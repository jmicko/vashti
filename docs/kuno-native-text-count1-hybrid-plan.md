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
release all-bin Clippy pass. Complete-model device acceptance and timing were
pending at this source checkpoint and are now recorded below. No installed service, consumer or release
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
columns, including nonzero surrounding intervals. Source gates pass; subsequent
complete B70 acceptance covers all4572 recurrent calls in127 steps, as below.

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
performance now pass for the fixed-public B70 cases below. No API, installed service or Vashti
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
requests are outside this experiment. Fixed-public B70 device acceptance and
timing now pass, as below; production acceptance remains separate.

## Complete-model shared execution acceptance, 2026-10-08

Source `775d985`, `fc1e89e`, `d351bd6`, `62a52ba` and `1a80d5b` completes the
operator-to-model transfer on the original mapped Ollama Flash-Next blob.
The selected v5 inline/intel96 artifact completes the75-prompt/128-output
fixture with all128 original native IDs,127 following steps in36280ms
(3.5006 tokens/s),65978ms prefill and3272ms bind. Seed424242/temperature0.
Its actual29205 CPU misses/31755 GPU hits compare with36916/24044 using
standard banks; the same inline artifact's standard sample reaches3.3473
tokens/s with48254ms prefill. These are single ordered samples, not statistical
or matched Ollama measurements. Prefill and host page behavior prevent treating
the larger bank as a universal speed improvement. The10-token/s target is unmet.

The exact artifact/source, all13 samples, outer validators and limits are in
[the aggregate receipt](../apps/kuno/benchmarks/text/receipts/next-shared-speed-20261008.json).
Normal builds through the batched-parameter step measure1.5190→1.5630→2.8263
tokens/s. A CPU-profiled sample is excluded from clean timing selection. The
explicit CPU-Q8 trial fails original128-ID validation and is not selected.
All accepted full samples pass the original normal/fresh/cancellation/poison
checks and full1378-owner/818413568-byte MAIN zero audits, source0 and no
context retirement. Existing runtime/private caps and actual swap0/core0 apply.

The final normal Cargo release build uses `--features direct-device`; no SDK,
driver or production dependency override. From the frozen v5 artifact metadata,
the protected runner invocation is:

```sh
python3 apps/kuno/benchmarks/text/run-native-count1-hybrid-lab.py \
  --cargo-artifacts /media/jmicko/faststore/kuno-development/next-shared-build-20261008-v5-inline/cargo-artifacts.jsonl \
  --model /media/jmicko/faststore/ollama/blobs/sha256-89059c880c2f000b24520919d4ba2edd2a8101327c7f8ab33b2280e01addda75 \
  --card intel --cpu-miss-cap 10 --cpu-workers pooled --cpu-gpu-overlap \
  --mapped-weights --large-weight-cache --compact-info --batch-tasks \
  --gpu-router --defer-router-io --retained-metadata --fast-gpu \
  --next-shared-kernels --next-shared-gdn --info-batch-upload \
  --device-dispatch inline --bank-profile intel96 --long-fixture \
  --run-id NEW_PUBLIC_RUN_ID --receipt /tmp/NEW_PUBLIC_RECEIPT.json
```

Replace both placeholders with fresh public-fixture destinations. Pin the
normal binary from source `1a80d5b`/SHA256`fe490699…` before running; Cargo's
mutable target must still match that metadata. Pause installed admission and
verify exclusive free-GPU access first; the runner does not isolate services
on its own. Its48-GiB user cgroup independently checks swap0/core0. Resume
admission and verify actual physical memory after completion.

Current source gates:119 core tests,3 ignored;479 normal/481 direct native
tests plus two additional offline shader suites each;50 imported runtime tests;
22 hybrid Python controls; strict all-target Clippy, formatting and provenance.
No arbitrary-request, quality/full-logit, long-context/soak, mixed-GPU or
production/native-default acceptance follows. No API or consumer changes;
required Vashti changes: none. The next speed bottlenecks are CPU expert work
and fine-grained GPU handoffs, before speculative expert placement/prefetch/MTP.

## Bounded dynamic cache-admission experiment, 2026-10-08

The user authorized continued speed work toward10 Next tokens/s. The earlier
cap10 path assigns every actual cache miss to CPU, so its GPU banks do not
admit new decode experts after prefill. This is intentional in that calibration
policy, not an adaptive expert cache. The next explicit fixed-public experiment
admits canonical integer CPU caps0 through10 instead of only0/10; defaults,
ordinary modes and installed routing do not change. Source10 remains the
stationary-bank control. Smaller caps send the first capped actual misses to
CPU and fill remaining misses into the original GPU bank with its existing
canonical victim protection and all-family publication. No CPU hit is moved
off GPU, no probability/rank is skipped, and no prediction or private history
is persisted. The policy remains fixed for a complete sequence.

Original ten-rank geometry, maximum40-MiB public source jobs,1-MiB CPU/private
host envelope, full original MAIN/IO budgets and private owning erasure remain.
The opaque proof binds the exact cap, actual miss set and unaltered source;
changing cap after classification rejects. GPU uploads remain on the original
parent stream and require the same source accounting, joined CPU consumers,
whole-model fault/finality and completed publication checks. The existing
explicit fast/metadata/shared/inline selections now accept any independently
pinned cap within this protected count1 mode. Other entry points, noncanonical
strings, booleans, negative or greater-than10 caps still reject before access.

Receipts pin requested/observed caps and actual CPU work, with a cap-specific
maximum of48*cap per step. Existing GPU hit/upload/read counters distinguish
stationary residency from actual cache adaptation. All original128 native IDs,
fresh/cancel/poison and full owning audits precede performance acceptance. A
smaller CPU fraction may lose to PCIe transfer; it is not selected by assumption.
This backend-only laboratory change requires no Vashti or API work.


### Separate CPU FMA candidate, 2026-10-08

`cpu-fma` is an off-by-default native-lab Cargo feature. The protected count1
runner must independently select `--cpu-fma`, pooled workers and original-float
activations. Actual Cargo features and the typed `cpu_fma_arithmetic` receipt
must agree; missing/mismatched flags or Q8 activation arithmetic reject before
inference. The feature's native entry point rejects other modes or a CPU
without AVX2/F16C/FMA. It selects `Backend::Avx2Fma` explicitly; normal CPU
backends, deployment builds and API choices are unchanged.

This candidate reconstructs original packed weights and uses fused products
plus32 accumulated lanes instead of the exact64-lane round-point policy. It
retains the same source/private limits, input/activation/partial owners,
physical16-worker plan, finite checks, cancellation, joined consumers and
wiping scratch. No private cache or new staging is introduced. Other supported
formats retain the existing original CPU decoder. Independent f64 row tests
bound forward error relative to the L1 product norm; synthetic whole-expert
error is bounded by the reference output infinity norm. Neither proves model
quality. The existing complete128 original output-ID comparison, full fences,
owning erasure and physical recovery remain mandatory before speed acceptance.


### GPU source page-cache advice experiment, 2026-10-08

The protected runner must explicitly select `--reclaim-gpu-sources`, mapped
weights and CPU cap10. At the model's existing opaque completed projection
boundary, inspect only committed GPU tensor tiles and expert bank slots.
Checked original tensor-relative extents are rounded inward to complete native
pages; the pinned read-only file descriptor receives POSIX_FADV_DONTNEED.
Linux64 except s390x is the implemented ABI. There is no global cache drop,
file write, mapping invalidation, new private staging or GPU mutation. Existing
mapped slices retain their immutable lifetime contract. The kernel may ignore
advice or retain pages used by mappings; no memory budget credit is granted.

`reclaim_gpu_sources` must match actual opt-in; `gpu_source_advised_bytes` is
bounded0..32GiB and positive only when enabled, and `gpu_source_advice_ms`
records pre-decode elapsed cost separately. These are numeric aggregate counters;
selected expert IDs/ranges are not persisted. Existing source identity, full128
output IDs, fences, owning zero audits and physical recovery still precede speed
acceptance. Other caps/unmapped input reject before device access. Installed
inference and the shared API are unchanged; no Vashti changes are required.

The motivation is a measured warm decode stall in source98df150:11601731584
disk-read bytes and61267 major page faults over127 following steps, with no
warm GPU weight uploads. This suggests competition between original GPU-source
file pages and CPU experts, but neither the hint nor the hypothesis is selected
as an improvement without complete physical acceptance and counter comparison.


### Explicit B70 128-slot profile, 2026-10-08

`intel128` is a separate fixed protected count1 profile, only admitted for Intel
hardware with the original48-layer2560/640/512 Q4_K gate/up and Q5_0/Q8_0 down
triples. All banks use two original packed64-slot shards. Original group/rank
maps and maximum80 private group/status owners remain unchanged; greater slot
indices are runtime data within the same bounded transient metadata. There is
no new expert quantization, tensor copy on disk, skipped rank or altered merge.
Normal profiles retain their original device ceilings and defaults.

Before allocation, verify every original triple and the exact24GiB bank page
charge. This profile requires an explicit5GiB static cache and31GiB reserved
limit (including the existing parent admission of private owners, public vectors,
source window and observed baseline). Keep the original512MiB residency margin;
actual allocator enforcement remains mandatory. Each segment remains at most
128MiB. Ordered uploads, batched legacy expert path and word-down remain excluded.
Both prefill and decode use these original banks; CPU cap10 remains static after
prefill. Full128 output IDs and all original owning/lifecycle gates still precede
acceptance. This does not assert that the total model fits in RAM or that the
expanded bank will improve performance. No API/Vashti/installed changes.


The first128 device launch rejects safely before bank allocation because its
544128-byte committed host ledger exceeds the prior512KiB reserve. The corrected
source reserves640KiB separately for original BankSlots, checks actual capacities
at every existing boundary and remains within the original12MiB total parent
host envelope (11206656 bytes accounted). Original2MiB staged MoE candidates
and1MiB CPU-private limit are unchanged. This is an explicit capacity charge,
not bypassing the failed guard. No performance acceptance follows that launch.


### Ordered full scratch clears, 2026-10-08

The explicit protected `--native-scratch-clear` choice uses the already checked
same-stream native buffer clear for complete original GDN/QSA/GR/coordinator
scratch owners. Only consecutive initial clear operations are batched, at most
16 owners/64MiB; public geometry beyond these limits retains the original shader
clear. Original stage producers still check cancellation and queue every consumer
in the same order. No status/output owner or inactive tail is omitted, no parent
fence is removed and the private arena's final owning erasure/audit is unchanged.
The existing physical full-handle, extent, same-stream and no-alias checks apply
before any command. The native clear closes earlier compute passes and schedules
earlier consumers before resetting scratch. This is an implementation dispatch
change, not permission to publish or reuse private data before completion.

`native_scratch_clear` must match independent opt-in; the numeric-only
`observed_native_scratch_clear_calls` must be positive when enabled (at most
1000000 for the bounded fixture) and zero otherwise. No request metadata enters
these counters or cache keys. Full128 output IDs, fresh/poison checks and original
owning audits plus physical recovery remain mandatory. No installed or API/
consumer change. Host gates pass486 native tests plus two offline shader suites,
26 Python controls and strict all-target Clippy before physical efficacy testing.


### Explicit residual CPU integer arithmetic, 2026-10-08

`--cpu-q8-residual` requires independent `--cpu-q8`, pooled AVX2 workers and
an artifact without `cpu-fma`. The original-float and ordinary single-pass Q8
choices remain selectable; there is no automatic failure fallback. It reuses
original Q4_K/Q5_0/Q8_0 weight codes across three Q8 activation passes: quantize
the original input per32 elements, then quantize each of two successive residuals.
All scales/quants/sums are call-local private owners, admitted by their actual
full metadata capacities before copying input and wiped by the existing joined
owning protocol. The original1MiB CPU envelope remains; no model requantization,
new persistent cache, copied GGUF or skipped expert/rank is introduced.

Integer horizontal sums stay in SIMD registers and Q5 original bit decoding is
vectorized, with the saturation proof still covering signed weight-128 and
activation clipping to[-127,127]. Float accumulation/reduction differs, so this
is explicitly approximate. Independent dequantized-weight/f64 original-input
row checks bound error by1e-5+1e-5 times the L1 product norm; complete expert
checks bound against the original portable result by5e-4+1e-5 times its infinity
norm. Existing single-pass integer tests include saturation edges. These tests
are not model-quality evidence. Full128 original token IDs, all complete-model
fences/owning audits and physical recovery still precede speed acceptance.

`cpu_q8_residual` and `observed_cpu_q8_residual` must both match actual opt-in
and the constructed operator. Q8 disabled/mismatched/ambient flags reject before
inference. No installed/default/consumer changes. Host gates pass124 core tests
(3 ignored),486 native tests plus two shader suites,27 Python controls and strict
all-target Clippy. Complete physical correctness and speed remain pending.


### Selected CPU source read-ahead, 2026-10-08

Protected `--cpu-mapped-prefetch` requires independently selected mapped weights
and pooled workers. Before launching existing joined consumers, it calls Linux64
POSIX_FADV_WILLNEED on each selected original gate/up/down packed expert span.
Existing inode, ownership, extent and read limits are checked; the mapping and
original bytes remain unchanged. No future routing is predicted, no model is
copied or requantized, and private jobs/inputs retain their original full-owning
admission and erasure. This advisory operation is not proof of residency or a
speed improvement. Installed inference and public APIs are unchanged.

The selected and constructed operator booleans must match independently pinned
runner flags; ambient flags are overwritten. Numeric-only warm advice counts
must equal three times completed CPU misses (zero when disabled). Counter reads
occur outside the measured process resource window. Mandatory original128 IDs,
fresh/poison and full-owning audits remain. Device efficacy is pending.


### Read-only mapped GPU source reclamation, 2026-10-08

Explicit protected `--pageout-gpu-sources` requires source reclamation and its
existing mapped static cap10 completed-boundary contract. For the same complete
original source pages retained on GPU, the current experiment releases the
original mapping's clean file PTEs with Linux MADV_DONTNEED, then applies fd
DONTNEED advice. This supersedes the ineffective COLD/PAGEOUT experiment. Only
the original immutable read-only regular-file map is addressed: future accesses
refault identical pinned inode bytes, including existing borrowed slices. It
never addresses anonymous/COW inference allocations, changes file contents,
unmaps/replaces the original map, or reopens its path. Original stat/inode,
foreign tensor/extent bounds and full-page interior restrictions remain. See
[Linux API semantics](https://man7.org/linux/man-pages/man2/madvise.2.html).
Receipt `gpu_source_mapping_advice` must independently match
`discard-clean-file-ptes` when enabled (`descriptor-only` for fd-only reclaim,
`disabled` otherwise). Advised bytes do not establish actual reclamation.

Ambient selection is overwritten. Receipts must match independently selected
pageout and source-reclaim flags. All original128 IDs, completed producer/consumer
fences and full owning audits remain required. No installed/API/default change.


### Original 144-slot profile in unchanged GPU page envelope, 2026-10-08

Explicit Intel-only protected `intel144` uses two72-slot shards per original
projection. For validated original Q4_K gate/up and Q5_0/Q8_0 down shapes, each
72-slot allocation still fits the same64/128MiB exclusive bins as64 slots.
Across48 original layers the conservative bank charge remains24GiB; the static
cache remains5GiB and device ceiling31GiB. No new allocator configuration, reduced
private budget, logical output or rank omission is introduced. Original shard
boundaries are now driven by one checked public capacity table;144 requires72+72,
128 retains64+64 and96 retains64+32. Foreign geometry rejects before allocation.

Actual original host ledgers are charged within the existing640KiB component /
12MiB parent envelope; errors retain original rejection and erasure. The original
1378 owners/818413568 bytes and complete private/source proof inventories remain.
Full128 original IDs and every owning gate still precede timing acceptance.
No installed inference or API change. Physical efficacy is pending.


### Explicit worker calibration and compiled two-pass integer candidate

Protected `--cpu-threads` selects4,8 or16 pinned physical workers;16 preserves
prior behavior. Nondefault counts require pooled workers and the constructed
operator count must match the numeric receipt. Original thread metadata maximum
and CPU1MiB private envelope are unchanged; smaller pools grant no extra budget.
The runner overwrites ambient values and rejects unsupported/boolean counts.

Separate compiled `cpu-q8-two-pass` uses one residual Q8 activation refinement
instead of two; it requires independently selected `--cpu-q8-two-pass --cpu-q8
--cpu-q8-residual` with pooled workers and an artifact without `cpu-fma`. Original
weights, all ten ranks, private full capacities and erasure remain unchanged,
including clearing the unused refinement array. The actual operator reports2
passes; standard residual reports3, disabled reports0. No automatic fallback.

This is approximate. The independent original-weight/f64 row bound includes the
mathematically derived group quantization term max|x| sum|w| /254^2 for two passes;
three passes retain the previous bound. Reconstructed inputs also independently
check that group error, including an8-epsilon floating round allowance. Complete
original-float expert checks bound error by1e-3+1e-4 times infinity norm. These
synthetic bounds are not model-quality claims. The original full128-token gate
remains mandatory and unchanged before any physical speed acceptance. No
installed/API/default change. Device efficacy pending.


### Original 160-slot profile, bounded three-shard capacity experiment

Protected Intel-only `intel160` uses the original72+72+16 packed shards, with
unchanged exclusive power-of-two allocator bins. Conservative bank charge is
27GiB; explicit static dense cache is3GiB and the device ceiling stays31GiB.
All original public-shape and original complete constructor capacity/admission
checks precede allocation. The smaller dense cache can evict/reload weights;
this tradeoff is measured rather than assumed to improve speed. It grants no
extra private budget and changes no rank/selection/output or installed policy.

One checked public shard table governs logical source extents, GPU bindings,
slot bounds and receipt inventory. Original inactive/whole-owner audits remain;
all three last-slot boundaries are tested independently. The resident launch
proof validates actual gate/up/down shard inventories against that same table;
missing or extra shards reject before dispatch. Original host ledger
capacity remains separately checked within640KiB / original12MiB parent.
Full128 original IDs, mandatory numerical probes and full private/source owning
proofs plus physical recovery still precede acceptance. Device efficacy pending.

### Reuse the existing CPU pool for overlapping orchestration

The pooled overlap operator currently creates a scoped OS thread at each
layer, even though arithmetic already runs in a persistent pinned Rayon pool.
The internal candidate uses that same pool's `in_place_scope`: the GPU parent
closure stays on the original caller thread, while one joined pool task
coordinates the original broadcast arithmetic. No cross-layer/request job
survives the scope. The original32KiB orchestration reservation remains
subtract-only within the original1MiB CPU envelope. The result/metadata stays
in the existing full-capacity wiping owner; errors, panic and cancellation
still join all CPU work before returning, and private Partials keep their
original wiping ownership. There is no scheduling, arithmetic, model or API
change. Existing overlapping success/failure tests, public numerical probes,
full128 original IDs and complete original owning/erasure gates precede device
acceptance and speed claims. Physical efficacy is pending.

### Buffered original CPU jobs, mapped GPU source retained

The independently compiled `cpu-buffered-weights` candidate requires matching
`--cpu-buffered-weights` selection, pooled workers and the original mapped
model, without CPU mapped read-ahead. GPU source borrowing stays unchanged.
Only actual selected CPU gate/up/down spans use the existing checked positional
read into call-local original public packed vectors. There is no alternate
file, complete-model copy, format change or persistent expert cache. Original
40MiB public capacity and1MiB private/header/worker bounds still precede fill;
actual capacities remain checked and all consumers join before the original
full-capacity wiping header owner drops. Each successful original span read
increments a RAM-only atomic counter; the numeric fixed-public warm receipt
must observe exactly three reads per completed CPU miss and zero when disabled.
No request-based identifier or diagnostic enters production persistence.

The hypothesis is that one sequential original-span read avoids parallel
per-row mmap fault stalls; its allocation/copy cost can instead worsen speed.
Original numerical results must be identical to the same arithmetic mode,
including all128 reference IDs, unchanged full owning/erasure/fresh gates and
physical recovery. This is an unaccepted laboratory hypothesis until measured.

### Warm-only GPU timing diagnostic

The existing full-lifetime profiler exhausted the unchanged private-I/O cap
before inference because it flushed each initialization dispatch's INFO page.
The revised explicit fixed-public `--kernel-profile` keeps timing paused through
startup, probes and prefill. It arms only the first following decode step,
pauses on every return/unwind before further inference or cleanup, and reports
exactly one profiled original native decode step. This changes diagnostic scope,
not timing acceptance: profiled runs remain excluded from clean speed selection.
Per-client shared atomic pause state changes no kernel arithmetic, existing
source/target ownership, scope/fence, admission limit or retirement rule. All
profiling allocations remain subject to the original I/O guard; any overflow
still terminally rejects. Original complete fixture/erasure gates are mandatory.


### Successful CPU operator host phase diagnostic

Only the fixed-public native lab enables RAM-only aggregate CPU host durations.
The four numeric counters cover successful-call preparation, parallel gate/up,
parallel down, and total through the final finite/tail/file guards. Clock reads
and counters are disabled on ordinary core operators. Joined CPU calls may
overlap GPU work; these intervals are not additive GPU/wall time. Quantization
and final guards account for the difference from the summed component times.
No ranks, pointers, weight spans, prompt/output values or request history are
retained; original private envelopes and all publication/cleanup gates remain.
Receipts subtract before/after counters at completed model boundaries.


### Shared vector projection with original finite fault fusion

Explicit shared kernels may now combine with `--fast-gpu-fault`. The same
original vector/subgroup arithmetic writes each projected row and ORs its
nonfinite fault in that writer. The last public row tile independently scans
every logical output-owner index outside the complete matrix span, preserving
the removed separate finite kernel's coverage without racing active writes.
All tiled active rows are covered by their respective writers. Logical owner
prefixes, strides, foreign stream/fault extents and complete matrix/tile bounds
reject before launch. Offsets/whole matrix span stay in existing privately
owned INFO, never pipeline keys. Only public geometry and last-tile selection
identify the opt-in kernel. No extra MAIN/source allowance or reduced physical
owner/erasure proof, automatic selection, installed/API/default change.
All seven original formats require numeric, inactive-suffix-NaN, active-NaN,
sticky fault and inactive-prefix-infinity startup controls; shared five-case
controls have independently checked markers. Host checks require the exact
logical output binding, and the shader uses actual dispatched group counts
for complete tail coverage on both16/32-lane subgroup sizes.
Unchanged128 IDs and original complete owning/lifecycle checks precede timing.

### Packed full-span runtime audit copies

Only explicitly selected batched-INFO laboratory streams pack the original
post-erasure Upload/Uniform copies into staging owners with at most1 MiB of
logical source coverage. A larger original owner stays alone. Original source
order and each complete API span are preserved at disjoint aligned offsets;
the last stage's power-of-two padding is cleared and fully inspected too.
The plan retains only at most256 pairs of numeric byte lengths (4 KiB), no
source offsets or contents. Original readable owners remain individually
inspected. Stage planning and full padding charge reject before first erasure
or verification copy if the unchanged hard budget cannot cover them. Each
actual allocated stage remains in the original ledger on partial setup, copy,
map, scope, nonzero/error/unwind, or final-erasure failure. All original sources
remain retained until both original erase passes, complete full-span inspection
and original scope/fence/publication checks succeed. The receipt counts actual
full original and packed staging bytes/buffers. Non-batched paths retain one
stage per source; no clean pool or cross-epoch reuse. This changes staging
allocation/map overhead; it removes no original source coverage or checks.
Source tests and the unchanged complete128-ID/owning/lifecycle device gates
are mandatory before timing acceptance.

### Synchronous GPU error scopes across existing scheduler batches

The explicitly batched-INFO WGPU laboratory may share OOM/Internal/Validation
scopes across one already-owned serialized task batch. Guards remain local to
the same synchronous thread/stack; none is stored in a Send server or survives
the batch call. Original task/resource/INFO ownership, order, compiler checks,
flush limits, nested scopes and both admission/error fences remain. Popped
futures enter the original checked error ledger before scheduler flush/readback
or publication. A failure of any enclosed call still rejects; same-class
per-dispatch attribution is not retained by a shared WGPU scope. A retired
stream stops the remaining batch, dropping its original wiping metadata.
Single-stream Interleave and Sequential schedules use an enqueue-many seam;
multi-stream interleaving keeps its original individual order and flushes.
The default backend method preserves individual enqueue; WGPU's non-batched
path also keeps original per-call guards. No new queue, host/private allowance,
API/default, unbounded wait, or early acceptance is introduced. Host scheduler
order checks, original LIFO/error tests and the complete128-ID/owning/lifecycle
fixture are required before accepting timing.

### Public kernel family labels for one-step profiling

The bounded first-following-step diagnostic distinguishes shared dense Q4K/
Q6K/Q5Q8/float, MoE gate/up and Q5/Q8 down, GDN and router families with static
names. These are public model/operator classes; labels contain no rows, source
spans, routing, request values or timestamps. Shader source/arithmetic and
compilation identity are unchanged. Profiled runs remain excluded from speed
selection and subject to the original I/O/full-model/owning checks.

### Batched actual-resource metadata checks

The selected retained-metadata laboratory may query each original MoE or hybrid
validation boundary as one ordered batch of full handles rather than one runtime
call per handle. Every original stream, offset, logical/full extent, dtype, shard
and physical alias check remains. No identity proof is cached across boundaries.
Cardinality is public and bounded (ten hybrid owners; at most sixteen MoE/private
and bank owners). Actual handle and returned metadata Vec capacities, plus the
fixed numeric extent array where present, must fit the existing 4 KiB host-header
allowance; no budget is increased. Non-retained resource lookup keeps its original
submission semantics. Original full-model and owning/lifecycle acceptance gates
are required before accepting a speed result.

### In-place exclusive allocator cleanup

Exclusive pool cleanup may retain pages in place rather than drain and rebuild
the complete page array at every submission. The same free-count increment,
aging threshold, reuse decrement, explicit release and storage flush remain.
Only retained pages moved by removal of preceding holes change descriptor page
indices. Actual handles and bindings still resolve to their original storage.
No cleanup cadence, admission, erasure, publication or cache budget changes.
Host controls must cover retained identities across repeated sweeps, multiple
holes, held bindings after original handle release, reuse aging and explicit
release, followed by the original complete device fixture.

### Word-sized full-span zero inspection

CPU post-erasure inspection may compare complete eight-byte words using safe
unaligned byte-array loads, then check every remaining tail byte. The supplied
view is still the exact complete owning span, including padding; no copy, GPU
read, narrowed prefix, early success or allocated scan buffer is introduced.
Nonzero data still rejects. Original view-length checks and map lifetime/cleanup
guards remain. Exhaustive host controls cover unaligned views, exact lengths,
all poison positions through word tails and untouched outside guards, followed
by original full device owning/lifecycle and output acceptance.

### Consecutive full scratch clears in one compute dispatch

The explicit protected count1 `--compute-scratch-clear` experiment requires
retained metadata, shared kernels, compact INFO and task batches, and excludes
`--native-scratch-clear`. Consecutive scratch-only clear lists may group two to
four distinct complete owning buffers in one compute shader without ending
the existing compute pass. Each binding covers the actual full underlying
buffer, including padding; every index is overwritten with zero. Actual arena
epochs, same streams, full handles, actual resources and physical aliases are
checked before launch. The existing 4 KiB scoped metadata allowance and16-owner/
64 MiB clear limit are unchanged; each group is at most64 MiB. Single owners
or groups beyond the byte bound keep original individual shader clears.

The pipeline key contains only fixed public binding cardinality, not indices,
routes or request data. No scalar INFO, private device allocation, persistent
proof, early completion or clean pool is added. GDN/QSA operation plans retain
all original clear events; the first event encodes the consecutive group before
any following consumer. MoE/GR/model scratch lists preserve original ordering
and all final erasure, scope, publication and whole-owning gates. Startup probes
independently poison complete spans of unequal sizes, verify every selected
byte and padding for groups2/3/4, verify unselected owners remain unchanged, and
reject duplicate IDs before encoding. Exact typed selected/observed counters,
probe markers, original output IDs and owning/lifecycle completion are mandatory
for receipt acceptance. This remains an internal laboratory option.

### Active-row dispatch bounds

GR, GDN and QSA batch consumers, plus active-status folds, may dispatch only
the already-validated active token rows1..8 instead of always dispatching eight
rows. Original kernel bodies, active arithmetic, runtime spans, full MAX8
allocations and all clear/erasure/ownership bounds remain. Omitted rows previously
performed only workgroup-local reductions or guarded no-op work; original status
and output stores are inside the same active-token condition. QSA causal token
spans, cache append order and fixed per-token selectors remain. This is dispatch
geometry only; no request value enters a compilation key or persistent trace.
The fixed-public full fixture, fresh continuation and original owning/lifecycle
gates are required before accepting speed.

### Ready-before-reuse grouped status folds

The explicit protected `--grouped-status` experiment requires retained metadata,
shared kernels, compact INFO and task batches. Three GR read statuses may be
folded together after mix and before its consumer; five GDN statuses at the
original tail fold boundary; eight QSA rectangular statuses after append.
Original QSA per-token causal folds remain individual. No status is reset or
reused, no fault is observed or published, and no consumer completes between
its former fold and the grouped boundary. The original event/cancellation plans
and completed fault/erasure/owning gates remain.

One fixed-public3/5/8-member shader evaluates every original active status word
and atomically ORs the same bit2 into the original sticky fault owner. It ignores
inactive rows exactly as before. Actual arena epochs, full owners, logical view
bounds, streams, offsets, physical extents and aliases must be checked before
launch. Actual handle/resource/binding/INFO capacities and scoped View metadata
fit the original4KiB host header; no private allocation or budget increase.
Private view coordinates use the original full-capacity wiping INFO ledger;
only public member cardinality identifies compiled kernels.

Independent device probes cover groups3/5/8, active counts1/3/8, unequal widths,
first/last active poison, deliberately poisoned inactive/padding words, sticky
fault preservation and duplicate-owner rejection. Exact typed selection and
observed-call counters, probe markers, all128 original model IDs, fresh
continuation and whole-owning/lifecycle acceptance are required before timing
selection. Default and installed inference remain unchanged.

### Shared GR RMS subgroup reduction

The explicit protected shared-kernel laboratory may use vectorized aligned
GR RMS loads and subgroup reductions for MAX8 active rows. Original max scaling,
epsilon placement, gamma multiplication, per-element finite output clearing,
whole-stream input validity and status values remain. Sum reduction round points
change; the original all128-ID gate is unchanged. Only public width specializes
the shader; counts, offsets, strides and epsilon remain in the original wiping
INFO ledger. No new global scratch allocation, retained private buffer or
increased host/device allowance is introduced. Parent full model-owner/physical
alias proof is unchanged; the narrowed seam validates all aligned logical spans
and streams before dispatch. Default/installed reference execution is unchanged.

Independent protected device probes cover the actual2560-wide/four-stream
shape, active counts1/3/8, finite input/gamma, NaN/infinity, invalid epsilon,
overflow-safe large finite inputs and untouched full inactive/padding spans.
Readback guards wipe on success/error. Original model output, fresh continuation,
poison rejection and owning/lifecycle gates remain mandatory. Exact bounded
warm shared-GR call evidence and the numeric/finite/status/owning probe marker
are required before accepting a timing receipt.

### Completed PLE full readbacks as one ordered group

The selected shared-kernel laboratory may submit the original OLD, NORMALIZED
and NEXT PLE full-owner reads as one three-member readback after the unchanged
whole-model fence. Full owner/epoch/physical proofs and every original finite,
inactive, padding and length validation remain; no write intervenes between
these immutable completed diagnostic sources. Raw controller ownership wipes
on every success/error path, and all raw leases drop before the same independent
PLE audited cleanup, second whole-model fault proof and all-family preflight.
No state commit or owner credit advances early. The original4MiB host component
cap must cover the simultaneously retained raw maps plus all converted float
vectors and existing history, with the original64KiB metadata reserve. Runtime
readback/full-audit reservations and original owning inventories are unchanged.
The unselected path retains separate reads. Original full-model128 IDs and
fresh/poison/full-owning/lifecycle gates are mandatory before timing acceptance.

### Factored original Q4_K vector dots

The selected Next shared dense and expert kernels may factor each original
Q4_K dot into nibble-vector/input dots and input sums multiplied by the same
original block scale/minimum. Packed byte accesses, bindings, supported shapes,
source owners, runtime INFO, all finite/sticky fault handling and full owning
spans remain. This changes floating round points without requantizing weights.
Original independent f64 row controls and complete128-ID/fresh/poison/owning/
lifecycle gates remain mandatory; no tolerance or output gate is relaxed.
The separate fully resident Ornith path keeps its current arithmetic.

### Count1 hybrid inactive-grid and immutable header reuse

The count1 hybrid merge dispatches its sole active row instead of eight rows;
the original kernel stores only when token==0, and the narrowed owning proof
already rejects every count other than1 before encoding. Full MAX8/MAX80 owners
and the following full-span finite/inactive/owner/map/status inspection remain.
Within each full inspection invocation the immutable checked owner header may
be evaluated once and reused, independently of subsequent local fault results.
Every header/bijection/word/status predicate is unchanged; no result survives
the invocation or becomes a shader/cache key. Independent malformed/full-tail
controls and original full128-ID/fresh/owning/lifecycle gates remain mandatory.
The factored Q4_K arithmetic is withdrawn after an accepted but slower trial;
shared Next kernels again use the original accepted vector dequantization.

### Full-capacity host erasure and direct guarded readback copies

The protected native laboratory may erase exclusively owned heap capacity using
aligned volatile u64 stores plus exact byte prefix/tail writes and the original
compiler fence. All capacity bytes remain covered, including uninitialized Vec
spare capacity, without reading them. Mapped erasure retains its existing API
and complete-span contract. Guarded readback copies reserve the complete logical
size before private content arrives, then copy directly into that owner without
a redundant zero initialization pass. No private-content growth, temporary Vec,
pooling, wipe omission or increased budget is permitted. Original conversion
error/unwind ownership, mapped leases, audits and final full zero proofs remain.

Host controls exercise all offsets0..7, lengths0..256, outside guard bytes,
empty/odd/aligned/32768-byte copies, source preservation and complete capacity
erasure. All original full-model IDs and owning/fresh/lifecycle gates remain
mandatory before selecting a timing. Installed execution remains unchanged.

### Explicit zero-only readback reuse laboratory

`--readback-reuse` requires protected batched INFO and selects native
`KUNO_LAB_READBACK_REUSE=1`; all other modes remain unchanged. This supersedes
earlier no-pool wording only for these Readback owners. At most16 exact-size
Readback owners, totaling8MiB, may remain after complete accepted inspected
two-pass erasure. Scheduling has drained, every returned/map lease is gone,
every audit map is dropped and the final clear/fence/error proof has completed
before transfer. Uniform resources and mapped Upload owners are never pooled.
Failures, non-inspected cleanup and live leases cannot introduce reusable owners;
retirement forbids future allocation. Every newly dirty reused owner again enters
the original full-owner ledger, two passes, full inspection and completion gates.

Retained zero-only bytes remain charged against the original64MiB runtime I/O
cap; activation transfers exact existing charge to the current Readback total.
All source/audit headroom checks include retained charge; there is no allowance
increase or early dirty-owner credit. Pointer storage is reserved before content
and bounded to16 entries within the original metadata envelope. Exact-size reuse
is internal to one stream/context; no external buffer accessor survives the
returned lease proof. All retained owners disappear on context destruction.

Four host controls cover strict selection, exact conserved charges, count/byte
limits, unselected behavior and failure atomicity. The runner requires exact
policy and bounded positive warm reuse evidence. Every original128-ID, fresh,
poison/cancellation and complete owning/lifecycle check remains mandatory.
No production/private worker or installed service change is implied.

The subsequent selected laboratory pool allows up to64 exact-size clean
Readback owners within the same8MiB cap. Actual preallocated pointer capacity
plus pool header must fit4KiB before any content. This supersedes the preceding
16-entry candidate ceiling; all original cleanup/charge/lease gates remain.

### Primary mapped Upload GPU erasure laboratory

Explicit `--gpu-upload-clear` selects `KUNO_LAB_GPU_UPLOAD_CLEAR=1`, requiring
protected batched INFO and actual native MAPPABLE_PRIMARY_BUFFERS capability.
Only dedicated Upload usage gains MAP_READ and COPY_DST. Public weights, MAIN
owners, Uniform and Readback allocation usages remain unchanged. The pinned
wgpu29.0.4 core directly maps a MAP_WRITE buffer at creation, rather than creating
its alternate hidden initialization stage; Vulkan's MAP_READ+MAP_WRITE location
remains CpuToGpu as with existing MAP_WRITE-only uploads. This is a capability
selected experiment, not a portable performance claim.

Both original full Upload erase passes become full GPU clear commands at the
same drained/scoped completion boundaries. Between them the original Upload
buffer itself is mapped READ, and its complete actual API-visible span must
be zero, using original guarded map callbacks/timeouts/full-word inspection and
unmap failure retirement. This direct CPU visibility proof replaces only the
Upload verification-copy stage; every Uniform still receives its original full
stage copy/padding check. Readback owners remain directly inspected. All maps
are dropped before the final full GPU erase/fence/error acceptance. No pass or
source byte is omitted; failed completion/inspection retains owners and retires
the context. Nonselected uploads retain original volatile mapped WRITE erasure.

The original64MiB cap and conservative Upload/Uniform duplicate preflight remain;
no owner is credited early or transferred to a pool. Exact policy/capability and
bounded positive warm clear evidence must accompany unchanged128 original IDs
and full fresh/poison/owning/lifecycle acceptance. No service/API/frontend change.

The all-size primary Upload clearing candidate is withdrawn from speed
selection after a deliberately stopped slow fixed-public run. No original
full-ID/lifecycle acceptance is claimed for that run; normal volatile Upload
erasure remains selected. The experimental option is disabled by default.

### Inline private Readback completion laboratory

`--inline-readback` / `KUNO_LAB_INLINE_READBACK=1` requires protected batched
INFO and the compiled inline device dispatcher. After original full readback
setup/submission, the calling thread performs the native device map wait instead
of waking a polling thread and parking on its callbacks. One bounded native
30-second map window starts before requests; all callbacks must be delivered
and accepted when Wait returns. Absent/rejected callback or timeout rejects and
retires through the same guarded leases/mutation scope. No second deadline is
started. The original async conversion future still owns every pending map and
returned wiping controller/lease; all original drop/unmap/conversion failure
handling, source/full-copy spans and sanitization remain. Nothing is published
early. Default async behavior and production inference deadlines are unchanged.

Exact policy and bounded successful warm inline-map batch evidence must accompany
all original128-ID/fresh/poison/full-owning/lifecycle gates before speed selection.
This changes internal scheduling only; no public API or Vashti change.

### Partitioned recurrent reduction laboratory

The existing opt-in shared GDN laboratory may split each original128-key
reduction into four fixed32-key partitions within one128-invocation workgroup.
Partial sums and validity masks live only in bounded workgroup memory. Summation
order changes; the independent original scalar numeric/invalid-column probes
and unchanged original128 generated IDs remain mandatory. All four partitions
combine old-state/prediction/delta validity before state updates, then combine
updated-state/readout validity before publication. Every invalid column clears
all four disjoint state intervals. Every lane follows identical barriers; no
workgroup scratch is reused before all preceding readers complete. Original
complete owners, status, full-span cleanup and lifecycle gates remain unchanged.
No allocation budget, production behavior, model blob, API or frontend change.

### Paired original expert-bank dispatch laboratory

The existing opt-in shared count1 MoE seam may bind exactly two original
contiguous shards in one gate/up dispatch and one down dispatch. The existing
parent physical proof still checks every original projection/shard identity,
origin, extent and alias. The narrowed seam additionally validates each segment
geometry/size/stream and their exact total capacity. Original runtime map and
rank slots choose the immutable weight binding, retaining all accepted packed
row arithmetic and rank-order merge. Public shard capacities specialize the
shader; request-derived routes remain runtime-only. Other shard counts retain
the original implementation. No bank copy/requantization, extra private owner,
cleanup omission or budget increase. Original128 IDs, owning/fresh/poison and
lifecycle checks remain mandatory. No API/frontend/installed behavior change.

### Explicit work-stealing CPU row dispatch laboratory

The separately compiled `cpu-work-stealing` laboratory feature requires
independent runner `--cpu-work-stealing`, protected count1 and pooled workers.
It replaces synchronous per-worker broadcast with joined scoped tasks over the
same bounded disjoint row chunks and full-capacity wiping header owner. Exact
original packed math, cancellation/error clearing and joined completion gates
remain. Physical worker pinning still occurs once during pool construction;
parent affinity is unchanged. All tasks finish before header inspection/drop,
including parent/worker panic and error. Opaque Rayon jobs/stacks retain the
same explicit runtime exclusion as the original broadcast. No persistent
request jobs, changed host budget, API or installed/default execution. Original
core all-format bit, failure/join/header/thread tests and whole-model128-ID/full
owning/lifecycle acceptance are required before speed selection.
