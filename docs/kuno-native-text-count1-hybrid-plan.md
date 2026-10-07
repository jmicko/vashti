# Count1 CPU-miss / resident-GPU plan

Read-only source proposal against root `4195238`, 2026-10-06. Root decides after
the protected CPU/GPU calibration evidence. This implements none of Stage 3 and
establishes no speed benefit. The scope is the existing fixed-public whole-model
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
