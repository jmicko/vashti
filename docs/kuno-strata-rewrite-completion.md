# Complete Strata Intel inference rewrite: completion gate

Approved 2026-10-14. This is backend engine work, not gateway/API/UI/release work.
The user requires a complete coherent Rust port before live speed testing. This
checklist is a source-to-source audit, not a sequence of throughput experiments.
Each row needs its connected consumer, real ownership/placement and state contract;
adding an unused helper or optional switch does not complete a row.

## Reference and execution scope

Pinned MIT upstream `fb58e0dbc8399662c0e47c76578c6e878b14f6cf`, checkout
`/home/jmicko/.codex/worktrees/kuno-native-split-gguf/strata-research-20261009`.
Follow its **Intel** single-device inference path: original full 512-expert
Qwen3.8-Flash-Next DASLab IQ3_S, original Q2 MTP sidecar, real router and target
verification. Intel B70 is the selected device. CUDA/HIP-only kernels, other model
families, web server/UI, distributed GPUs, optional content-persisting conversation
caches and benchmark programs are not required to reproduce this inference path.
Do not use those scope boundaries to omit prompt prefill, sampling, state commit,
CPU/GPU residency, or a dependency of the working measured Intel path.

Rust must own model loading, graph execution, request state and inference kernels.
Generated WGSL/SPIRV and Rust/ash Vulkan are the explicit backend substitution for
SYCL kernels/queues. No foreign inference graph/library is linked. Preserve MIT and
GGML notices. Substitution must preserve buffer placement, data movement, grouping,
work sharing, ordering and state transitions; an alternative schedule must be
recorded as a remaining gap until reviewed, not silently described as parity.

## Connected dependency checklist

Current state is **incomplete**. Entries below are preliminary source findings;
all need a final audit of actual connected dispatches, not just file names.

| Component | Pinned upstream | Current Rust connection and required completion |
| --- | --- | --- |
| Artifact geometry, packing and tokenizer | `sycl/src/core/weights.cpp`, `src/io`, native expert layout, program model setup | Rust GGUF/tokenizer/binding exist. Audit format-specific load/conversion, chat-template and sampling behavior; original files remain shared/read-only. |
| Exact dense placement and expert budget | `sycl/src/program/generate.cpp` model/cache setup; `sycl/src/core/expert_cache.cpp::open_sized` | Exact dense roots exist. Native global byte-prefix planning and an owning bounded GPU loader now compile; graph validation consumes the real artifact plan. Experimental core construction now invokes the sole global owner before prefill, and the core48-layer coordinator selects global routing/HOST fetch/role execution. Adaptive exchange/private remap updates, native format conversion audit and physical/numerical acceptance are unfinished. The ordinary feature-off engine retains uniform banks. |
| Low-RAM host complement and promotion | `generate.cpp` source/adaptive tier; `sycl/src/core/verify.cpp` fetch/rebase | Original host complement, direct exchange and GPU selected fetch exist. Audit full byte ownership and asynchronous safe publication against upstream; do not duplicate the whole model in RAM. |
| GPU route/group planning and CPU contribution | `verify.cpp` router, `resident_plan`, doorbell/pool/host plans; `src/kernels/cpu` | Real GPU router exists. Audit the selected working Intel all-GPU-fetch versus host-pool mode and reproduce its actual mode, including overlap/merge where active. |
| Native expert math and grouped rows | `sycl/src/kernels/cuda/iq_kernels.dp.cpp::row_entries`, `native_gu_kernel`, `native_down_kernel` | Packed integer math/BDA roots exist. Separate GU roles, 4/2/1 chunks, activation, intermediate storage and downstream consumers are now selected together in the Intel core; all 158 WGSL/SPIR-V variants validate, including 105 physical-pointer checks; numerical/model execution acceptance remains pending. The global core now uses one per-role launch containing pinned row_entries4/2/1 loops, scalar IQ4_XS, native signed integer finish and XOR reduction; numerical/model execution acceptance remains pending. Historical fixed-chunk kernels are retained separately. |
| GR write/read/norm/down/up/injection | `sycl/src/kernels/cuda/fused_gr.dp.cpp::fused_gr_read_multi`; `verify.cpp::gr_read_group` | Native pending-write norm, sliced BF16 down/injection, fixed slice reduction and up/mix/commit are now connected and unconditionally selected in the core, including explicit PLE/final boundary writes. Resident original weights and the enlarged private scratch owner are charged together. Structural acceptance passes; numerical/model execution acceptance and final substitution audit remain pending. |
| GDN projection/conv/gates/recurrence | `verify.cpp`, `cuda/native_gdn*.dp.cpp`, `cuda/fused_gdn.dp.cpp`, `cuda/verify_kernels.dp.cpp` | Core now connects native BF16 alpha/beta window projection, conv/L2, controls, double-buffered four-row-group FMA recurrence/output norm and state-only accepted-prefix replay. See the GDN receipts below. Optional Q8 producer fusion, prompt-path fidelity, physical execution and final numerical/queue audit remain open. |
| QSA projection/norm/RoPE/KV/indexer | `verify.cpp`, `cuda/native_qsa*.dp.cpp`, `cuda/qsa*.dp.cpp` | Core now connects FP16 KV for current and committed rows, native fused norm/RoPE, FP16-rounded raw index keys, retained pooled blocks, native Intel scores, 64-cell attention/merge and rejection clearing together. See the QSA receipt below for precise storage/scheduling substitutions and remaining numerical/state acceptance; do not claim full QSA fidelity yet. |
| PLE source rows, projection and history | `verify.cpp`, `ngram`/`ple` kernels, PLE table reader | Original grouped reads/hash/history exist. Audit asynchronous deduplicated source reads, grouped BF16 projections and ordered history snapshots/commit. Never persist request n-grams. |
| Prompt prefill and streamed experts | `sycl/src/prefill/prefill.cpp`, `gemm.dp.cpp`, MoE MMQ/Intel implementations | Rust bounded graph prefill is not full prompt-path parity. Port connected chunk budget, streamed layer staging and dense/expert batches, including prompt history consumed by MTP. |
| MTP loading/prefill/draft/head | `src/core/mtp.cpp` as built by Intel target, Intel kernels | Core connects MAX8 front, packed native FP16 cache, dense attention and BF16 FF/head GR to real draft/repair consumers. Original Q2 experts still need the native scaled activation/integer path; router/shared and queue/capture fidelity, numerical/state acceptance, arbitrary input and supported sampling remain required. See the front and rest receipts. |
| Suffix proposal and draft policy | Intel speculation implementation and `DraftPolicy` | Request-local Rust suffix/cost controller exists; audit supported window lengths and coupled sampling rather than fixing to four public-fixture rows. |
| Target verification and accepted-prefix commit | `sycl/src/core/verify.cpp` checkpoint/commit methods | Real greedy verification/state-only repair exists. Audit all active windows and acceptance cases against GDN/QSA/PLE/positions; independent correctness oracle remains missing. |
| Graph capture, queues and synchronization | `verify.cpp`, session/mtp graph construction, Intel queue/event implementation | Native Vulkan owned recording/replay exists. Audit stable complete recordings, transfer dependencies, scratch lifetimes and actual overlap; a sequence of independent Wgpu launches is not equivalent. |
| Request lifecycle and arbitrary input | upstream CLI/session/inference entry points | Current Rust public-fixture gate is not a general engine. Connect bounded arbitrary prompts and request-local sampling/history, with no persistent content, protected memory and orderly erase/close. No gateway/UI work required. |

## Validation and the speed gate

1. Keep the connected port selected as one engine configuration. Fix compile,
   shader, numerical and ownership bugs as they arise. Do not retain a partial
   upstream behavior as default-off merely because an unfinished run was slower.
2. Interim GPU execution is allowed only to prove correctness/completion, with
   swap/core dumps disabled and actual resource recovery checked. Disable operator
   profiling. Do not compute/present a throughput ranking or run comparative speed
   trials during this stage. Fixed public fixtures may be used; private content
   never enters receipts/logs.
3. Final completeness audit must identify each actual call path, source revision,
   resident owners, dependent consumers and every resolved substitution. Remaining
   upstream inference dependencies mean the rewrite is incomplete.
4. Then run end-to-end correctness/completion tests: prompt prefill, count1 and
   supported multirow verification, acceptance/rejection/repair, EOS and capped
   generation, and recovery. Correctness failure blocks performance acceptance.
5. Only after (3) and (4) pass, run live throughput comparisons on the assembled
   engine and fix performance bottlenecks. 30 accepted tokens/s remains the desired
   result; a speed number does not itself prove completeness or correctness.

No frontend changes, release or installation is authorized by this checklist.
Historical artifacts/receipts remain evidence, not the next direction. If root
credentials expire, document where the connected implementation stands and stop
at a safe boundary instead of spending time on privilege workarounds.

## Current implementation block

`strata_expert_roles.rs` connects gate/up/DOWN 4/2/1 chunks to the same authentic
physical source roots and one HOST fetch sequence. Gate/up borrow two 640-value
planes from each existing 2560-value PART row; the activation kernel consumes and
zeros both planes before original DOWN preparation/consumption. Per-role stamps
use the existing status owner and are restored for DOWN. All intermediate owners
remain in PrivateArena; no extra private weight/input roots are allocated. Eight
extra INFO owners/layer are charged together. Core selection is unconditional;
there is no speed-driven fallback. Other incomplete audit rows remain required.

The protected runner now requires `--execution-check` and rejects operator
profiling while the rewrite is incomplete. Its source receipt labels the run as
execution/correctness only, never throughput acceptance. Release compilation, strict core Clippy, fmt/whitespace, runner syntax, the
158-variant shader set and original direct-device regression (1,346 passed,
1 ignored) pass. Runner refusal of speed/profile invocations was exercised
without root, service calls, output creation or GPU access. No GPU inference run was performed in this
implementation block. Full model numerical execution remains pending.

## Connected GR implementation block (2026-10-14)

Runtime source commit `56663a3`. `strata_gr.rs` / `strata_gr.wgsl` port the pinned Intel `gr_norm_split_port_kernel`,
`gr_down_sliced_kernel`, `gr_down_reduce_kernel` and `gr_up_multi_kernel` as one
sequence. Normalization uses one workgroup per token/stream, native XOR reduction
order, weighted values retained per lane and pending residual FMA. Sliced projection
shares 128 activation columns, computes 320 down and four injection rows using the
original resident BF16 matrices, and reduces the 80 slices in fixed order. Up/mix
shares the reduced low-rank values and commits the pending residual output.

`Prepared::combine` defers the write into the next GR read. Pending source/output
identities and active count must match; unfinished pending writes cannot pass the
whole-model consumer audit. PLE at layer 1 explicitly flushes the preceding write.
A compact final-head read flushes all prompt rows first, preserving prompt/MTP
history; no-head prompt chunks also flush before their completed proof. Previous
injection gates are retained in the existing private normalization-status owner
before the next injection projection overwrites them. The active core uses that
owner as 64 F32 words; its physical allocation stays unchanged.

The existing UP owner becomes the 80x8x324 partial workspace, growing from 524,288
to 1,048,576 bytes. The complete model quote and GR family construction quote
both add 524,288 bytes. No extra private root, decoded weight copy or per-decode
weight upload is added. Actual downloaded model headers confirm GR down/up/inject
are original dtype 30 (BF16) with dimensions [10240,320], [320,10240], [10240,4].
Native GR writers join nonfinite faults directly; retired pointwise status-fold
dispatches are not falsely credited. Ordinary feature-off code retains its original
allocation, arithmetic and immediate-combine path.

Explicit Vulkan substitutions: pending writes target the existing ping-pong
residual owners; core scratch uses the existing private UP owner rather than
Strata's queue-global partial allocation; LOW global stores have one actual writer
instead of identical racing writes from every up workgroup. WGSL exp/inverseSqrt
replace SYCL native math. These are documented implementation differences, not
proof of numerical parity. Read-only dummy bindings for absent injection/pending
inputs do not alias a same-node writable owner.

Validation logs in `/media/jmicko/faststore/kuno-development`:

- `strata-connected-gr-build-20261014.log`: release build passes.
- `strata-connected-gr-final-clippy-20261014.log`: strict core Clippy passes.
- `strata-connected-gr-shaders-20261014.log`: 163 WGSL/SPIR-V sources pass,
  including 105 physical expert-pointer checks and all five GR kernels.
- `strata-connected-gr-original-regression-20261014.log`: original direct-device
  regression passes (1,346 passed, one ignored).
- `strata-connected-gr-core-layout-tests-20261014.log`: 12 GR ownership/layout
  tests pass in core mode. No GPU or model execution is involved.
- `strata-connected-gr-core-regression-20261014.log`: full core unit run is
  **not green**: 1,148 pass, 200 fail, one ignored. Representative failures still
  require the old 144-expert geometry, original MTP allocations, old host cap and
  pre-port target budgets; test-scope/contract reconciliation and triage remain
  required. Do not claim all failures are harmless without further evidence.

No live GPU inference, operator profile, speed trial, installation or API/UI change
was performed. Full rewrite and numerical/state acceptance remain unfinished.
Continue the complete source-to-source checklist; do not benchmark or disable the
GR sequence to select an intermediate throughput winner.

## Global byte-prefix and owning-loader block (2026-10-14)

Pinned Intel `generate.cpp:4336` sizes native-pack slots from the **global ranked
prefix in bytes**, using each layer's original blob size and 256-byte alignment.
It stops at the first pair that exceeds the budget. Copying a uniform number of
experts per layer changes that design even when the total byte budget matches.

`strata_global_plan.rs` implements that policy from actual `BoundGraph` tensor
metadata, validates the embedded full `STRP` ranking/reverse table, and assigns
every original expert to exactly one resident or HOST source. With the existing
240-slot-equivalent budget of 23,574,528,000 bytes, the actual pinned artifacts
select 11,570 ranked pairs / 23,573,836,800 bytes; layer counts range 160..342.
These are public header/profile calculations, **not throughput measurements**.

The resident layout is one logical original gate/up/down blob arena in rank
order, represented by bounded physical chunks in Vulkan. Expert blobs never
cross a chunk boundary; chunk offsets and role offsets are explicit. The public
startup upload window is at most 16 MiB, rather than a whole-model RAM mirror.
Missing experts have separate bounded original role pieces in HOST memory. Empty
tiers allocate no invented weight roots. `source()` resolves the authoritative
resident chunk or HOST piece for every (layer, expert, role). The startup reader
copies original GGUF ranges into one chunk and zeroes its padding; no dequantized
model or on-disk conversion is created. Plan instance/binding checks keep source
capabilities tied to one original graph and layout.

`strata_global_gpu.rs` adds the owning cache loader. The executor verifies model
provenance, precharges all persistent roots, and scopes exact allocation through
the existing admission mechanism. The owner verifies full physical roots, distinct
backings and byte totals; role borrowing cannot claim HOST experts as GPU resident.
Allocation errors poison the owning admission. This constructor is **not yet
called by the inference coordinator**: it must replace legacy banks, never add
a second 23.6 GB cache alongside them. Its GPU load/closure has not been executed.
Graph validation currently runs only the host-only plan check.

Required connected continuation (no intermediate speed trials):

1. Replace core ordinary-bank construction with this single global owner and
   its actual complete precharge. Preserve the original feature-off engine.
2. Import the planned HOST complement and connect the GPU fetch/stage path to
   original blob/role addresses; retain real source owners and recover them on
   errors. Do not force unequal counts into old fixed 120/120/128/128/16 extents.
3. Connect per-expert physical pointer tables, GPU routing/group plans and all
   private arena/stage budgets. The native backend must retain and track the
   actual BDA source owners even when descriptor bindings do not list every
   global chunk. Do not use fake buffers, invented residency or loose lifetime
   assumptions to satisfy the old 15-source layout.
4. Connect adaptive exchange/map publication to the same global layout, with
   varying blob sizes and joined reader/writer boundaries. Correct commit and
   full private erasure remain mandatory.
5. Then validate execution/correctness of the connected path. Remaining full-port
   checklist rows still block any speed trial or completion claim.

Strict selected core Clippy and fmt/whitespace pass. Two host-only allocation
tests pass: independently checked pinned prefix/counts, complete 512-expert source
coverage, exact per-role bounds, and empty/partial/full cache cases. Logs:
`strata-global-owner-final-clippy-20261014.log` and
`strata-global-plan-final-tests-20261014.log` in `kuno-development`. No model
GPU load, private-request execution, speed/profile, service or installed change.
The preceding 200 core-suite failures remain unresolved; these tests do not
supersede that acceptance gap.

The authorized sudo refresher was renewed to one bounded 240-minute helper at
`/media/jmicko/faststore/kuno-development/strata-global-port-sudo-20261014.pid`;
the preceding helper was stopped only after verifying the new process. Recheck
the process/cache before protected work, and stop cleanly if required root access
is unavailable. Do not infer cache validity from this note alone.


## Global HOST, source owner and native read dependencies (2026-10-14)

Source commit `47d330f`. Full rewrite remains **incomplete**; no GPU inference or
speed/profile execution took place in this block.

`strata_global_host.rs`, a child of the existing Vulkan HOST owner, imports only
`Tier::Host` pieces from the new global plan. Resident pieces are explicitly
excluded from both the RAM envelope and construction, so the complement is not a
second full model mirror. Pieces keep their original plan indices; missing tiers
allocate no dummy buffers. A one-MiB reader copies each original expert role into
fully initialized nonlocal HOST-visible Vulkan allocations. Actual requirements,
heap placement and byte charges are checked before allocation. The full RAM quote
adds conservative global metadata/owner capacity (32 MiB), the immutable registry
(1 MiB) and bounded setup windows (two 16-MiB GPU upload windows plus the separate
1-MiB HOST reader). The CPU reservation is distinct from GPU admission; caller
proof of available/protected process memory remains required, not inferred from a
64-GiB numerical ceiling.

The original HAL backing, immutable registry authority and exact detach receipt
are used together. Ownership is recorded before the HAL import and before
registration; unknown exposure/import/detach/completion never grants backing-free
permission. Successful detach joins real readers and destroys registered buffers
before CPU zero/audit/free. Global page counts use their actual registry limit,
not the old fixed 432-page inventory. `strata_global_gpu::Cache::load` drains each
moved original chunk/upload before reading another, so queued Vec payloads cannot
silently accumulate the whole GPU cache in RAM.

`strata_global_sources::Sources` owns one plan, GPU cache, HOST complement and
48 exact 32-KiB pointer tables. Each initial (layer, original expert, role) entry
contains its actual buffer address plus offset, original role word extent and
resident/HOST flag; unused tail entries are zero. Same-stream full-resource
metadata, disjoint physical roots, bounds, alignment and plan identity precede
publication. Per-layer root dependencies cover every source used by that layer.
The executor's new unsafe `prepare_strata_global_sources` precharges all cache
and table allocations through the existing exclusive persistent admission and
owns the completed Sources object. It rejects previously constructed banks,
expert copies, other caches and duplicate construction. Ordinary bank paths reject
a simultaneously installed global owner. Closing drains the same client, drops
table/read aliases and detaches HOST before cache release; a failed close cannot
retry its way into a false safe-release claim.

**The inference coordinator does not call this constructor yet.** No global cache
was physically allocated, imported, executed or recovered. Dead-code annotations
make that gap explicit. The live experimental core still dispatches its old uniform
banks and old 15-source table; the installed service was not changed.

Native BDA consumers now have `KernelArguments::read_dependencies`. These are
actual retained buffers, not added shader descriptors: scheduler ordering includes
these streams, queued tasks own the resources, native nodes include SHADER_READ
hazards, recording identities include every dependency extent/address, and captured
recordings retain them through the actual fence. Dependencies require the owning
native stream and a registered physical expert program, with bounded capacity;
ordinary/non-native execution rejects them. Immutable external stream validation
also covers hidden dependencies. Physical lowering rejects writable gate/up/down
roots. An opaque allocation ID supports owner lookup without exposing mutable
memory locations or providing allocation/detach authority. Empty dependency lists
preserve ordinary launches. No API/protocol changes.

Validation logs under `/media/jmicko/faststore/kuno-development`:

- `strata-global-sources-build-20261014.log`: core release binary builds.
- `strata-global-sources-final-clippy-20261014.log`: strict core Clippy passes.
- `strata-global-sources-shaders-20261014.log`: 163 CPU WGSL/SPIR-V validations,
  including the existing physical-pointer variants, pass; no GPU dispatch.
- `strata-global-sources-plan-tests-20261014.log`: three pure layout tests pass,
  including empty/partial/full HOST reservation coverage. This caught and fixed
  accidental resident-piece admission into the HOST complement before execution.
- `strata-global-sources-original-tests-20261014.log`: ordinary direct-device
  regression passes, 1,349 passed and one ignored. This is CPU/unit coverage,
  **not** acceptance of the experimental core's whole-model state/math/closure.
- Rust formatting and `git diff --check` pass. The earlier full core suite's
  200 failures were not rerun or declared resolved by these results.

### Next connected implementation block

1. Replace native coordinator bank construction **before initial prompt prefill**
   with the sole global Sources owner and actual combined CPU/GPU admission. The
   old probe attaches HOST after prefill, which is too late. Audit native quant-grid
   versus old address-table construction: do not retain an unused legacy address
   allocation and quietly claim the same table quote.
2. Replace fixed 120/120/128/128/16 root bindings and old virtual-slot assumptions
   together in routing/group planning, selected HOST blob fetch/rebase, native
   separate-role GU/activation/DOWN and their session/consumer accounting. Use real
   Sources bindings/read dependencies; no fake weight buffers or invented permits
   to satisfy historical audits. Audit the native T=1..8 policy and GROUP staging
   capacity, not only the old four-row public-fixture path.
3. Pinned `sycl/src/core/verify.cpp:1259` shows resident/device planning;
   `:1387` couples selected HOST `fetch_blobs` with `rebase_ptrs` when PCIe mode 2
   is active. The grouped expert consumers need both, plus the real shared/CPU
   merge behavior for the selected Intel mode. Don't switch to direct remote reads
   as an unrecorded substitution simply because constructing pointers is easier.
4. Connect joined adaptive exchange to the same unequal layer counts/physical
   cells. Current global tables encode **initial physical cells**. Do not write
   request-derived recency/expert maps into publicly initialized MAIN roots and
   omit their erasure. Retain immutable physical-cell pointers plus a separately
   arena-owned private logical remap, or supply an equivalently proven private
   owner/publication/erase contract. Keep original model data separate from request
   history; swaps of different-size cells must not overrun allocations.
5. Reconcile coordinator permits/closure and exact family/root budgets with those
   actual dispatches. Then continue every remaining row of this checklist,
   including GDN/QSA/PLE, arbitrary prompts, complete prompt prefill, MTP graph and
   target/state commit. Correctness/completion may be checked; speed/profile stays
   blocked until the full connected completeness audit and acceptance pass.

Canonical checkout was untouched; work remains in `codex/strata-rust-port` in
`/home/jmicko/.codex/worktrees/kuno-image-speed/vashti`. Inherited untracked Python
bytecode is preserved. No installation, service/reset/rebind, gateway/API or Vashti
change. Goal remains active; sudo unavailability remains a stop/checkpoint rule,
never a completed-rewrite claim.


## Connected global core dispatch block (2026-10-14)

Source commit `d24c348`. This replaces the preceding block's uncalled-constructor
state: the protected language probe now calls
`ResidentModel::prepare_global_sources_before_prefill` after model/workspace
construction and **before** the first prompt chunk. The sole global cache/HOST
owner and real family stages are initialized together. `Workspace::run` selects
`run_global` unconditionally in `strata-intel-core`; it does not fall back to old
banks if global construction/dispatch fails. Old post-prefill HOST attachment is
skipped in this mode to prevent allocating a second cache.

`strata_global_moe::Workspace` connects real GPU top10 routing and selected-softmax,
private logical-expert remapping/group union, native Q8_1 preparation, selected
HOST fetch, gate/up, activation, DOWN and weighted reduction. Full48-layer private
completion stamps are read/accepted after the existing whole-model fault proof.
Real GDN/QSA/PLE pending acceptance and state commit remain connected in the copied
core coordinator. It does **not** synthesize old bank/session/frozen-source permits
or credit their counters. Existing ordinary feature-off driver remains separate.

`strata_global_expert_kernels` ports pinned `iq_kernels.dp.cpp:1784` row_entries:
one native dispatch per role runs4-entry chunks, a2-entry remainder, then a1-entry
remainder. IQ4_XS follows the scalar-only path (`Multi<23>::has` is false). The
integer finish expressions in IQ2_S/IQ3_XXS, integer scale weighting for IQ3_S/
IQ4_XS,32-lane XOR reduction and IQ4_NL's two16-value dot calls follow the pinned
source. Quantization retains the native single Q8_1 plane with FP16 scale. These
are source/structural findings; mathematical/model parity is not established by
shader acceptance alone. Older three/four-row arithmetic stays separate.

Selected original HOST roles are copied once per actual group, with per-role
GPU stamps checked before reading their rebased stage address. Actual immutable
cell pointers remain public setup metadata; the logical remap lives in PrivateArena
and starts as a complete per-layer identity permutation. Both actual source roots
and stage roots participate in native recording retention/read hazards. Static
stage-pointer metadata owns one additional exact32-KiB allocation. No route,
recency, prompt, output or private remap is uploaded into a public address table.

The new family owns nine small private roots plus three128-MiB stages. Constructor
allocation requests and model/private quotes account for the actual family;
stages cover the worst80 selected role cells. Source requests include48 cell
tables plus the one stage table. The old IQ-table constructor no longer allocates
48 unused legacy address tables in core mode. Historical MoE workspace/fixture
reservations still exist in model construction and need cleanup/reconciliation;
they are not credited as global execution.

### Validation and remaining limits

Logs in `/media/jmicko/faststore/kuno-development`:

- `strata-global-coordinator-build-20261014.log`: core release build passes.
- `strata-global-coordinator-final-clippy-20261014.log`: strict core Clippy passes.
- `strata-global-coordinator-shaders-20261014.log`:179 WGSL/SPIR-V shader variants
  pass, including physical lowering for all new original expert formats, routing,
  fetch, native row-entry helpers, quantizers, activation and reduction. No GPU.
- `strata-global-coordinator-original-tests-20261014.log`: final current-source
  ordinary direct-device regression passes:1,349 passed, one ignored. This is
  CPU/unit coverage of the ordinary path, not model/GPU acceptance of the core.
- Formatting and whitespace validation pass. Previous200 experimental core-suite
  failures are unresolved; no claim that the full core suite is green.

**No model/GPU execution, speed/profile trial, installation, service change or
reset/rebind occurred.** No physical global allocation/recovery or numerical/full
inference completion claim is justified yet. The complete rewrite remains open.

### Required connected continuation

1. Implement adaptive exchange/recency on these same global cells, and publish
   only arena-owned private remap changes after actual joined reader/exchange
   boundaries. Preserve complete per-layer bijections and original role extents;
   never overwrite public pointer tables with request-derived history.
2. Replace the language/speculative controller's historical frozen-bank receipt,
   activation and dispatch-policy assumptions with real global-family contracts.
   The current old controller can reject **after prefill** because these old
   counters/scopes are deliberately not fabricated. Do not loosen those old
   validators or claim this is a working end-to-end generator yet.
3. The family/router/quantizer/operator geometry handles T=1..8. Verify the
   coordinated GDN/QSA/PLE/final-head/target-row and checkpoint paths support
   those same windows; copying an eight-row expert kernel alone does not prove
   whole-target eight-row support. General bounded prompts/sampling/MTP request
   entry points still require connection/acceptance.
4. Audit the actually working Intel reference mode's CPU share, shared-stream
   overlap and transfer mode. This new core currently selects a single-queue,
   all-GPU HOST-fetch route; it is not proof of every upstream CPU/GPU plan.
5. Audit native format conversion/packing and byte budgeting against
   `weights.cpp`/native layout. Current global roots use original GGUF packing and
   original role-byte counts. This preserves source files and decoding capability
   but must not silently be called byte-for-byte parity with a converted native
   blob layout. Original buffers, pointer selection, role math and physical
   ownership now connect; all declared backend substitutions still need review.
6. Continue the full checklist's remaining GR/GDN/QSA/PLE, prompt prefill, MTP,
   graph/queue and state-commit rows. Full completion/state/numerical/recovery
   acceptance precedes all speed comparisons. Do not disable this connected core
   based on incidental timing or return to uniform-bank behavior.

Canonical checkout is untouched. Changes are backend internals only; required
Vashti/API changes: none. Inherited Python bytecode remains untracked/preserved.
Goal stays active and incomplete. The bounded sudo refresher was still verified
live during source work; revalidate process/cache before any protected execution.


## Connected global adaptive exchange (2026-10-14)

Source `e1dc698`. The global core now invokes adaptation after real48-layer
GDN/QSA/MoE/PLE acceptance and commit. This supersedes the prior unconnected
adaptive row; physical execution and full rewrite acceptance remain unproven.

`strata_global_adapt.rs` ports pinned `generate.cpp:12040` selection: every four
completed windows, missing experts with usage >=2 pair with their own layer's
least-used resident experts; gain must be >=1.5. Candidates sort globally by gain
and at most96 swaps survive. Usage decays by0.7. Unlike the historical cache,
there is no uniform240-resident assumption, per-layer two-slot cap or extra
recency grace. Equal-score ties use deterministic expert/layer IDs rather than
upstream's unspecified equal-score std::sort order. All resident physical cells
are eligible. The proposal keeps full per-layer logical↔physical bijections and
checks staleness before CPU commit. Discarded private candidates are wiped; the
best96 are retained after each layer to keep the proposal heap bounded.

GPU routing records logical expert usage in an arena-owned F32 root, preserving
native increment/decay semantics. The family/root quote adds128 KiB for that
root; the separate CPU cache/proposal/publication reservation adds2 MiB to both
native host quote and limit. Usage, remap, scores, proposals and offsets are
transient. No prompt/output/route/recency data is persisted or printed.

`Sources::exchange` authenticates every role's exact original GPU/HOST capability,
range and graph before mutation. `strata_global_exchange.rs` verifies owning HOST
registrations, actual buffer identity, nonoverlap, equal role extents and device
alignment, then exchanges original weights in place on the owning queue. Evicted
weights occupy the incoming experts' former HOST cells; no file read or full model
RAM duplicate is added. All readers join before exchange, and actual exchange
completion precedes private metadata publication. A pending mutation rejects new
source consumers. GPU remap/usage writes join and the CPU proposal commits before
that guard is lifted. Failed/unknown exchange/publication does not resume the
context; ordinary joined cleanup remains available.

Explicit backend difference: the blocking Vulkan kernel exchanges corresponding
original role words in place, rather than Strata's D2H exchange staging followed
by H2D and RAM-buffer commit. It preserves complete two-way contents and joined
publication; it does not implement opt-in asynchronous adaptation/refill-stream
or pipelined-window overlap. These remain documented substitutions for review.
Current ticks are completed core chunks, including prompt chunks; the next driver
port must reconcile adaptation/usage boundaries with actual prompt versus verify
phases. No implicit claim of exact upstream prefill scheduling is made.

Validation under `/media/jmicko/faststore/kuno-development`:

- `strata-global-adapt-build-20261014.log`: core release build passes.
- `strata-global-adapt-final-clippy-20261014.log`: strict core Clippy passes.
- `strata-global-adapt-shaders-20261014.log`:179 WGSL/SPIR-V variants pass,
  including updated private usage routing and retained physical expert kernels.
- `strata-global-adapt-policy-tests-20261014.log`: three host-only tests pass:
  global96 gain selection across unequal/nonprefix/empty/full tiers, exact
  eligibility/gain/staleness checks, and12 repeated paired-exchange simulations
  verifying every logical expert still resolves to its original weights.
- `strata-global-adapt-original-tests-20261014.log`: ordinary regression passes,
  1,352 passed, one ignored. No GPU dispatch or actual Vulkan exchange acceptance.
- Formatting/whitespace checks pass. Prior full-core failures remain unresolved.

No GPU inference, speed/profile, physical residency/recovery, installed binary,
service, reset/rebind or API/Vashti change. Goal active and incomplete.

Next: replace historical frozen-bank controller/receipt/activation assumptions
with actual global contracts, reconcile prompt/verification adaptation ticks,
and connect complete1..8 target/MTP/prefill/state paths. Complete packing/Intel
CPU-share/stream/queue and remaining checklist audits before speed tests. Keep
this entire cache+remap+execution path selected while fixing correctness; do not
fall back to uniform banks or choose an intermediate timing winner. Canonical
checkout and inherited untracked bytecode remain untouched.


## Connected global request and verification controller (2026-10-14)

Source `9abc0fd`, isolated `codex/strata-rust-port` worktree. This supersedes the
old language-controller handoff: the core now calls `strata_global_controller`
after constructing the real global sources and before any historical frozen-bank
receipt path. Feature-off public experiments remain separate and unchanged.
The whole rewrite is still incomplete; no live model or speed run occurred.

The connected controller uses actual target rows and teacher prompt tokens,
retains the final prompt carry, builds genuine MTP windows or suffix proposals,
checks their positions/bindings, and verifies every proposed token with the
actual target. A target rejection or EOS selects the accepted input prefix.
`LatestTargetRound` checkpoints and commits/repairs actual GDN/QSA/PLE state;
MTP repair then uses the accepted target carries. The controller checks target
and draft cursors before the next window and at final output. Accepted tokens
can be delivered through a transient callback. The protected public harness
currently discards that callback output and writes numeric metadata only.
It no longer needs old bank-fill, legacy source counters, or a legacy HOST
attachment to claim completion. It does not fabricate their receipts.

Global authority now checks the real sole source owner, attached stage/table
roots, active admission and absence of legacy banks/caches. Completed native
boundaries bind that owner to the actual graph and reject pending source
publication. Shared GDN/PLE/status/head selection uses an explicit global
execution scope rather than pretending frozen banks were attached. Core
construction reserves its coupled four-row/PLE/status capabilities. Historical
integer/frozen close-count constants are excluded for this path; actual full
PrivateArena/source erasure, retirement checks and joined executor/HOST close
remain mandatory. No private close check is bypassed.

Usage collection now happens only during genuine verification, including its
provisional rows, as in native window routing. Prompt chunks and rejected-prefix
state replays do not add usage. Every four *finished verification windows*,
after actual target acceptance/repair, triggers the existing paired exchange
and private remap transaction exactly once. A failed verify/repair/publication
retires the round instead of resuming a partially published cache. This resolves
the prior prompt-chunk adaptation scheduling gap. Native background ranking /
refill overlap is still an explicit unresolved backend difference.

Core request geometry can charge arbitrary nonempty prompt lengths with1..512
outputs and prompt+output extent<=4096. Its real permit must match the requested
prompt/output geometry before prefill. Historical feature-off fixtures keep their
exact lengths/caps. This is an internal bounded controller capability: the
current public CLI/tokenizer registration still chooses compiled fixtures,
not arbitrary production text. Greedy decoding only is connected here. General
sampling and production entry/lifecycle remain open.

Validation logs under `/media/jmicko/faststore/kuno-development`:

- `strata-global-controller-final-build-20261014.log`: core release build passes.
- `strata-global-controller-final-clippy-20261014.log`: strict core Clippy passes.
- `strata-global-controller-shaders-20261014.log`:179 shader variants pass offline,
  including the updated verification-only private usage router.
- `strata-global-controller-core-controller-tests-20261014.log`:14 selected
  core host-only tests pass, including the new prefix acceptance test over every
  compiled width/rejection position/EOS and retained global cache tests.
- `strata-global-controller-geometry-tests-20261014.log`: bounded core request
  geometry/lifecycle test passes, including extent overflow and cap rejection.
- `strata-global-controller-ordinary-regression-20261014.log`:1,354 pass,
  one ignored, no failures. Formatting and diff whitespace checks pass.

These are build/shader/host-only proofs, not whole-model numerical/state or
actual GPU exchange/recovery acceptance. Prior full-core suite failures remain
open. Default core still compiles four-row capacity; `strata-eight-window` and
its coupled GDN/QSA/PLE/MTP/head/registration policies require reconciliation,
not just a flag change. Old unused MoE construction owners/reservations remain.
Prompt prefill is still bounded3-row scheduling rather than native MMQ parity.
Audit native packing, CPU-share/transfer queues, GDN/QSA math and KV format,
PLE async reads, MTP graph/state, full graph recordings, sampling and general
input before declaring completeness. Keep the assembled global path selected;
no intermediate throughput trial or timing-driven fallback.

No GPU/service/install/API/Vashti/release/reset/rebind change. Canonical checkout
and inherited bytecode remain untouched. Next continue the coupled1..8 model /
MTP / commit and native placement-fidelity blocks. Goal active and incomplete;
speed gate remains closed.


## Connected eight-row consumers and native Q6 head (2026-10-14)

Source `ac6169a`, same isolated `codex/strata-rust-port` checkout. This supersedes
the preceding four-row-capacity limitation **structurally**, not as a claim of
GPU/model numerical acceptance or complete rewrite. The core now selects capacity8
unconditionally. Historical feature-off four/eight experiments remain separate.

The common capacity drives target logits/packet owners, host/runtime quotes,
checkpoint token arrays, compact packet layout and MTP proposal width. Core logits
and packet owners are8 MiB each. Allocation and acceptance guards use the same
capacity. Native dense projections now prepare and consume1..8 actual Q8_1 RHS
planes; GDN/QSA grouped projections and private stage borrowing follow that width.
Non-Q6 dense consumers reuse code loads across two vec4 groups for5..8 rows.
The selected core GDN streaming recurrence runs every supported width rather than
falling back to its different chunk arithmetic for non-four verification windows.
GDN checkpoint admission now checks streaming policy for every count>=4.

`strata_native_q6.wgsl` ports pinned `iq_kernels.dp.cpp` Fmt<14> /
`vec_dot_q6_K_q8_1`: native32-lane iqs mapping, original signed per16 scales,
integer dot*scale before float conversion, native Q8_1 scale addresses and XOR
warp reduction. It reads the original packed Q6_K roots and shares each code load
across actual RHS rows. No decoded weight copy is created. The core head uses one
prepared input plane set and the same complete16 original weight tiles. Real
prepare/tile/member counters credit only completed enqueue sequences. Selection
occurs after authentic global source/stage attachment, before the first prompt
input, so prompt prediction and target verification use the same head arithmetic.
Legacy Q6 project vectors stay explicitly max4 and cannot silently consume an
unsupported wider span. Explicit WGSL FMA is a backend arithmetic choice pending
GPU/native numerical acceptance. Other dense formats' arithmetic audit stays open.

Original Q2 MTP accepted-cache repair is now supported in the core, with1..8
accepted rows and bounded genuine queued-forward cursor spans. It preserves the
confirmed predecessor cell and replaces provisional tails using actual target
carries. Full Q2 conditioning/KV source verification, fault join, ledger and
cursor publication remain in the existing joined repair implementation. Core
construction uses the full draft vocabulary, removing the historical English-only
subset; its close validator uses that actual full-vocabulary owner plan. The old
Q8/fused-four experimental wrappers keep their separate contracts. Actual8-row
Q2 repair dispatch-budget/state execution remains a GPU acceptance gate.

Validation under `/media/jmicko/faststore/kuno-development`:

- `strata-global-eight-final-build-20261014.log`: core release build passes.
- `strata-global-eight-final-clippy-20261014.log`: strict core Clippy passes.
- `strata-global-eight-shaders-20261014.log`:195 offline shader variants pass,
  including native Q6 widths1/4/8, eight-row dense consumers and packet/physical
  expert paths. No GPU device is created by this validator.
- `strata-global-eight-q6-tests-20261014.log`: two core host tests pass. Nine
  synthetic Q6 blocks cover each column exactly once through native iqs mapping
  and match the independent canonical GGUF decoder; RHS scratch bounds reject
  short owners and unsupported widths. This does not execute the GPU dot kernel.
- `strata-global-eight-state-tests-20261014.log`: two host tests pass, covering
  every queued window/accepted prefix for1..8 MTP repair and all compact packet
  widths, including nonzero inactive-row rejection and complete owner quotes.
- `strata-global-eight-prefix-tests-20261014.log`: two selected tests pass,
  including target rejection/EOS at every compiled width and the retained global
  profile-prefix/complement test.
- `strata-global-eight-final-regression-20261014.log`: ordinary regression passes,
  1,358 passed, one ignored, no failures. Formatting/whitespace checks pass.

No live inference, benchmark/profile, GPU exchange/recovery, service/install/API,
Vashti, release, reset or rebind change. Canonical checkout and inherited bytecode
preserved. Earlier full-core suite failures are not resolved by narrow passing
coverage and require reconciliation. Eight-row **whole-model** acceptance remains
unproven despite connected source/owner/consumer coverage.

Next continue native CPU expert share and transfer/shared-work/queue overlap,
original-versus-native packing and remaining GDN/QSA/KV/PLE/prefill/MTP graph,
sampling and arbitrary input audits. Prompt scheduling remains bounded3-row
rather than native MMQ parity; QSA quantized KV and complete graph recording still
need work. Keep the complete assembled core selected while resolving correctness;
no partial timing winner/fallback. Goal active, full rewrite incomplete, speed
gate closed.


## Connected native CPU expert share and shared-work overlap (2026-10-14)

Source `e43e271`, same isolated `codex/strata-rust-port` checkout. The previous
all-GPU-HOST-fetch limitation is now superseded structurally. Whole-model GPU
correctness/recovery and the full rewrite remain unproven; no speed gate change.

Pinned `expert_source.cpp::expert_pool_dispatch_multi` chooses distinct experts
in real routing order, keeps resident experts on GPU, and sends the last
floor(missing*pcie_num/256) missing groups to GPU staging. The new core uses the
native base0.55 share, rounded to141/256, bounded to64 staged groups. GPU groups
are resident-first then transferred groups, while CPU jobs retain original rank
and token indices. The actual GPU router's private plan is read and authenticated;
no expert IDs or predictions are forced. Every original route occurs exactly
once. Private remaps still identify initial physical cells after two-way exchange.

`cpu_expert::native` is Rust-only: Q8_K signed-maximum/RNE activation quantization,
IQ2_S/IQ3_XXS/IQ3_S/IQ4_XS integer grid/scale chains, Q8_0 and the distinct native
Q2_0 CPU activation rules, IQ4_NL/Q2_0 DOWN, grouped gate/up/SILU and DOWN phases.
It follows Strata's pinned GGML3cf03257f219afbe7334045ff7c6a06ac68c627d generic
quantizer/dot equations and original public codebooks. AVX2 widens signed i8 values
to i16 and performs exact i32 sums; unsupported hardware uses scalar integer
arithmetic, not failure-driven backend selection. Up to eight RHS share decoded
public codes per row. Persistent Rayon workers split all group rows, and all
request inputs/quants/activations/partials/masks have wiping owners. No model
file read, decoded weight cache or full model RAM duplicate is added.

The owning global HOST complement now supplies sole scoped readonly CPU maps
of actual selected original physical role cells. Binding/plan/layer/phase/range
and actual registrations are checked. Noncoherent allocations invalidate after
joined publication. Sources are Arc-owned while scoped CPU readers run; mutation
and close require unique ownership, and every CPU/GPU-parent success/error joins
before maps leave scope. GPU expert/shared commands are actually flushed for
submission while the CPU pool works. Shared preparation does not consume routed
output early; the CPU merge and real GPU reduction precede shared finish and GR
combine. A layer-pending state prevents duplicate enqueue/finish publication.

CPU results occupy one extra1 MiB PrivateArena root; the complete request model
quote adds that root and16 MiB bounded CPU host reservation. The merge copies
only authenticated CPU-mask rows, retaining genuine GPU stamps for other ranks,
then original GPU probability-weighted reduction joins every actual partial.
GPU/source/CPU routes and returned CPU masks must agree. Private CPU outputs are
never written into public model backing or persisted.

The original paired exchange now records COMPUTE_SHADER/SHADER_WRITE ->
HOST/HOST_READ visibility before its joined fence. The vendored Vulkan encoder
helper changes no ownership/layout/allocation. Fence completion alone does not
satisfy this memory-domain operation; see the primary Vulkan
[synchronization guide](https://docs.vulkan.org/guide/latest/synchronization_examples.html)
and [specification](https://docs.vulkan.org/spec/latest/chapters/synchronization.html).
CPU mappings can overlap later GPU *reads*, never source exchange writes.

Validation under `/media/jmicko/faststore/kuno-development`:

- `strata-global-cpu-build-20261014.log`: release core build passes.
- `strata-global-cpu-final-clippy-20261014.log` and
  `strata-global-cpu-library-clippy-20261014.log`: strict engine/library Clippy pass.
- `strata-global-native-cpu-tests-20261014.log`: three host tests pass: native
  signed max/tie/Q2 rounding, all six formats versus independent canonical GGUF
  weights with quantized inputs, exact SIMD integer sums and scoped eight-row
  sparse-rank outputs/parent-error join. No GPU or source mapping execution.
- `strata-global-cpu-core-split-tests-20261014.log`: host-only native split test
  covers every1..8 width and resident/missing split, proving each rank appears
  once. No live router or transfer execution is inferred.
- `strata-global-cpu-shaders-20261014.log`:196 offline shaders pass, including
  the private CPU merge and retained native physical-pointer paths.
- `strata-global-cpu-ordinary-regression-20261014.log`:1,359 pass, one ignored.
- `strata-global-cpu-library-regression-20261014.log`:141 pass, two ignored.
- Formatting/whitespace checks pass. Previous full-core failures remain open.

Explicit differences/gates: Vulkan uses audited per-layer route/input readback
and one queued private CPU result upload, rather than Strata's mapped plan sink /
CPU-written mapped result rows and synchronization flags. Rayon row tasks and a
scoped CPU controller differ from the native parked/spinning physical-core pool;
current workers use available logical topology and are not pinned. Native default
share is fixed here; bandwidth-probed fraction, asynchronous refill/layer pipeline
and complete stable native recordings still need audit/port. Integer equations
are implemented, but optimized native multi-token floating reduction order and
whole-model CPU/GPU numerical/state acceptance are not proven by the canonical
host comparisons. Do not claim full Strata performance parity or correctness.

Next complete native packing/placement/stream/recording audits and remaining
GDN/QSA/quantized KV/PLE/native prefill/MTP graph/sampling/general-input paths,
then whole-engine correctness/recovery acceptance before any live speed trial.
All new CPU/GPU/cache/driver parts stay selected; no intermediate timing winner
or fallback. No GPU inference/profile/reset/rebind, service/install/API/Vashti or
release change. Canonical checkout and inherited bytecode preserved. Goal active
and incomplete; speed gate closed.

## Connected FP16 QSA family (2026-10-14)

Source `d5d4e49`, same isolated `codex/strata-rust-port` checkout. Pinned Intel
`generate.cpp:435` defaults to **FP16** KV; optional INT8/Q4/hybrid modes are
not an inferred requirement for this default path. `strata_qsa.rs` and WGSL
translate the connected native norm/RoPE, indexer and attention family. The core
selects them unconditionally for every real width1..8; the ordinary feature-off
engine keeps its original F32 path. Scalar F32 consumers reject the packed core
owners rather than misinterpreting half pairs.

- Current candidates and committed tokens store both K and V as actual packed
  half pairs. Attention consumes those exact rounded values before append, so a
  speculative prefix does not see a different precision from committed context.
  Logical tokens contain256 K words,256 V words and128 raw-index F32 words.
- Raw index keys first round through FP16, as native SET_ROWS/GET_ROWS does.
  Pooling reads only newly completed blocks; old source reads are bounded to the
  previous three raw rows. Completed normalized/RoPE pooled blocks persist in
  the actual stage POOLED owner. They are copied to the window's causal views;
  new blocks publish only after native norm/RoPE. No old block is renormalized.
- Fused norm uses the native256-thread,32-lane XOR reduction and grouped warp
  sums, weighted values and two FMA RoPE outputs. Simple RoPE uses the native
  host F32 theta scale and device F32 angles. Existing graph binding rejects
  unsupported scaling overrides. Intel index scores use four contiguous values
  per lane, XOR sums and head-wise nonnegative accumulation, without the old
  extra square-root divisor. Original chronological selection remains connected.
- Attention executes64 selected cells per chunk, two KV groups and twelve query
  heads/group. It shares query/KV loads, uses the native eight-value partial/XOR
  score reduction, per-chunk maxima/exp-sums and sequential FMA value accumulation.
  Native max-rescaled chunk merge and the original sigmoid gate feed the actual
  output projector. There is no F32 softmax/value consumer fallback in the core.
- Checkpoint/receipt identity and physical-root checks include the retained
  POOLED owner. Rejected-prefix repair clears only discarded packed cache rows,
  positions and pooled blocks from the accepted incomplete-block boundary onward.
  A strict bit-zero fold precedes cursor publication; accepted prefix pages stay
  intact. All owners remain in the private arena and erase at request close.

The complete quote follows the actual new owners: each stage's admitted pages
change36,306,944→19,529,728 bytes; the single shared MAX8 scratch bank changes
17,825,792→22,052,864 bytes. CAND shrinks to32 KiB, native attention partials use
8 MiB, and max/sum stats use64 KiB. This is source/host allocation accounting,
not a device memory or speed measurement. INFO still fits the existing bounded
per-stage reservation, including the two retained-pool dispatches.

Remaining substitutions/gates: packed cache is token-major rather than native
paged/head-major; bounded raw-index history remains in the private cache instead
of native three-row tail plus dead/spare owners. Final partial-block selection is
explicit rather than the native dead-key score/bias route. Pool candidates are
window-owned copies; native multi-query block-score load sharing, tail snapshots
and full queue/capture equivalence still require audit/porting. WGSL math replaces
SYCL native math; subgroup/FMA/norm/attention numerical parity and physical state
repair/recovery have **not** been run. These are visible remaining gaps, not a
claim that merely rounding an F32 cache completes FP16 storage parity.

Validation logs under `/media/jmicko/faststore/kuno-development`:

- `strata-qsa-build-20261014.log`: release core build passes.
- `strata-qsa-clippy-20261014.log`: strict release core Clippy passes.
- `strata-qsa-shaders-20261014.log`:206 offline WGSL/SPIR-V shaders pass,
  including all ten new QSA kernels. No GPU execution occurs.
- `strata-qsa-core-family-tests-20261014.log`:26 QSA family tests pass.
- `strata-qsa-math-tests-20261014.log`: two host tests pass: every causal width,
  packed/scratch/tail interval and accepted-prefix extent, plus an independent
  global F64 softmax oracle for the prescribed chunk merge, including2051 cells.
  The host oracle does not establish device arithmetic parity.
- `strata-qsa-ordinary-regression-20261014.log`:1,361 pass, one ignored.
- `strata-qsa-core-broad-audit-20261014.log`:58 pass, **two fail** in the wider
  `qsa` filter: old language/cache selection gates and a whole-model quote fixed
  at826,638,336 instead of the assembled1,275,363,328. They remain untriaged core
  compatibility/coverage work; the previous full-core failures remain open too.

Fmt/whitespace checks pass. No live model, device correctness/recovery or speed
run, service change, installation or Vashti/API change. Full rewrite remains
incomplete and the speed gate remains closed. Next coupled source block is GDN:
read `verify_kernels.dp.cpp`'s actual multirow convolution/L2 and four-row-group
FMA recurrence/output-norm, including its accepted-prefix state replay. The old
Rust streamer currently decays before its dot and places query scaling before
recurrence, unlike native; port their dependent producers and repair together.

## Connected native GDN producers, recurrence and repair (2026-10-14)

Source `351e29b`, same isolated checkout. `strata_gdn.rs`/WGSL translate pinned
Intel `verify_kernels.dp.cpp::gdn_conv_l2_multi_kernel` and
`gdn_step_norm_multi_kernel` as a coupled core family for widths1..8. This
supersedes the preceding receipt's old GDN arithmetic description.

The128-lane convolution reads each row's four causal RAW inputs from the original
three-value ring plus projected QKV. Native SiLU division and XOR/four-warp L2
produce unit-normalized Q/K and unnormalized V. Final RAW ring storage is kept
separate. Query scaling is applied after recurrent readout, as in Strata; the
old producer's pre-recurrence scale is absent. Both producer status planes join
the existing fault proof. Original alpha/beta projected rows feed the native
20-threshold softplus/negative-A and sigmoid control equations.

Recurrence uses the native512-lane block: four groups,128 columns and32 register
state rows per thread. Each group accumulates32 un-decayed state/key FMAs; the
four partials add in the prescribed order. Decay then enters delta, and the
updated state/readout use native FMA order. Query readout scaling, four-warp
head RMS, original gamma and sigmoid gate are fused before the actual output
projector. Existing separate candidate-state roots retain the untouched original
state. Retained inputs now contain the new unit Q/K and exact native controls.

Accepted-prefix repair uses the same native recurrence arithmetic in a separate
state-only shader. It reads the real retained prefix and pre-window state/ring;
convolution restores RAW history. No attention, FFN/router, head or output graph
runs. The existing finite-state/status fold, completed fence and cursor commit
remain required. Legacy recurrence/pointwise consumers are selected only outside
core; native core does not silently fall back on an intermediate speed report.

No new MAIN weight/state owner is added and all existing quote/retained-input
roots stay admitted. Native dispatches replace multiple old operators within the
existing INFO cap. Kernel launch checks actual hardware capacity for the512-lane
block; unsupported device capacity rejects rather than issuing an invalid block.
Physical pipeline/device-limit acceptance is still untested.

Explicit remaining substitutions/gates: candidate state stays head-major in the
existing Rust private pages rather than native row/head-major storage; speculative
and count1 state both use separate next-state roots rather than native register-
only/self-commit distinctions. K/Q are loaded into one shared plane per token,
not native double-buffered next-token preload. Conv and step scratch layouts differ;
old ACT/RECURRENT pages remain reserved but are not falsely credited as producers.
WGSL exp/inverseSqrt and compensated positive log1p replace SYCL math. The generic
original BF16 alpha/beta projector needs a dedicated native AB arithmetic audit;
optional Q8 output fusion and prompt/capture/queue fidelity also remain open.
Do not claim complete GDN/whole-engine parity from these source connections.

Validation logs under `/media/jmicko/faststore/kuno-development`:

- `strata-gdn-build-20261014.log`: release core build passes.
- `strata-gdn-clippy-20261014.log`: strict release core Clippy passes.
- `strata-gdn-shaders-20261014.log`:210 offline WGSL/SPIR-V shaders pass,
  including conv, controls,512-lane step/norm and state-only replay.
- `strata-gdn-core-family-tests-20261014.log`:17 family tests pass.
- `strata-gdn-math-tests-20261014.log`: host four-group FMA recurrence/readout
  agrees with an independent F64 outer-product oracle through widths1..8; every
  accepted prefix reproduces the corresponding sequence prefix. This is not
  device or whole-model numerical/state acceptance.
- `strata-gdn-ordinary-regression-20261014.log`:1,362 pass, one ignored.

Fmt/whitespace checks pass. Previous broader/full-core failures remain open.
No GPU inference, model execution, speed test, reset/rebind, service change or
installation. Full rewrite is still incomplete and the speed gate stays closed.
Continue the coupled checklist: native AB/dense/packing audits, original prompt
prefill/PLE/MTP initialization and full graph/capture/sampling/arbitrary-input
paths; reconcile those consumers before final completeness and physical tests.

## Native GDN AB and next-token preload (2026-10-14)

Source `3ae92cf`. Core controls now use actual original resident BF16
`ssm_alpha.weight`/`ssm_beta.weight`, not the generic projector's F32 arithmetic.
The executor authenticates model provenance, dtype30, actual2560-column geometry,
matching original output rows and tensor roles before borrowing existing cache
handles. No decoded duplicate, per-request weight upload or new MAIN owner.

The kernel follows `verify_kernels.dp.cpp::gdn_ab_multi_kernel`: two consecutive
rows per64-thread group,32 lanes, eight contiguous BF16 weights per pack, stride32
packs, per-element FMA and XOR reduction. Each weight pack feeds all real1..8
rows. Alpha adds original dt and native softplus/negative-A; beta uses sigmoid.
An additional Vulkan status join combines actual alpha/beta stamps before the
existing control-status fold. This status-proof dispatch is the documented
backend addition; neither a fabricated geometry nor a missing half can pass.
QKV/Z keep the original quantized bundle. The retired standalone core control
producer is removed, and bypassing the native AB owner rejects instead of
falling back. Old ALPHA/BETA scratch pages remain reserved, not falsely consumed.

Recurrence and state-only replay now use the pinned two-plane Q/K schedule.
Current inputs initialize before the loop; row group0 preloads next K and group1
preloads next Q into the alternate plane while current state math executes. The
next iteration's barrier publishes those writes. The state-only shader omits Q
loads. Shared memory grows by1 KiB/block, with no global owner/quote change.
This supersedes the previous receipt's generic AB and single-plane gaps; other
GDN/whole-port substitutions and physical acceptance remain required.

Validation logs under `/media/jmicko/faststore/kuno-development`:

- `strata-gdn-ab-build-20261014.log`: release core build passes.
- `strata-gdn-ab-clippy-20261014.log`: strict release core Clippy passes.
- `strata-gdn-ab-shaders-20261014.log`:211 offline shaders pass, including
  actual AB, its stamp join and both double-buffered recurrence variants.
- `strata-gdn-ab-math-tests-20261014.log`: two host tests pass. New BF16 test
  checks exact pack decoding/column coverage, every real RHS and inactive-row
  preservation against an independent F64 dot oracle. Existing native state/
  prefix oracle passes. Neither proves device arithmetic or synchronization.
- `strata-gdn-ab-ordinary-regression-20261014.log`:1,363 pass, one ignored.

No live GPU/model/correctness/recovery or speed run; no service/API/UI change.
Earlier full-core failures and the full completion gate remain open. Goal active.

Next coupled port: MTP front/prompt initialization. Actual pinned
`MtpDrafter::record_front(T,-1)` gathers and normalizes embeddings, projects
embedding Q8 and wide hidden-norm streams in groups≤8, broadcasts into HC
residuals, runs native fused GR, projects Q8 K/V, normalizes/RoPEs K and appends
in the actual KV format. `prefill` queues grouped front-only graphs and completes
one final fence. Rust global controller currently loops target prompt chunks of
three and calls scalar `ResidentMtp::append_kv` once per teacher row; accepted
repair also loops its front calls. Wrapping that loop with a batch name is not
parity. Implement real grouped owners/producers/consumers and state publication
before changing the controller. Native default hidden norm is wide
(`mtp_hnorm_stream=false`); Rust's one10240-wide norm already matches that
choice. Preserve teacher token/carry alignment and the prompt_len-1 MTP cursor.

## Connected native MAX8 MTP front (2026-10-14)

Source `147cd9a`, isolated `codex/strata-rust-port`. `strata_mtp_front.rs`/WGSL
add a real model-owned MAX8 front workspace rather than a scalar loop named as
batching. It follows pinned `MtpDrafter::record_front/capture_prefill`: original
IQ4_XS embedding rows, native1024-thread weighted embedding/wide-carry norm,
Q8_1/native dense FC embedding and hidden streams in groups≤8, broadcast into HC
residuals, native sliced BF16 GR, shared mixed-input Q8 planes for K/V, fused key
norm/RoPE and grouped cache append. Every real row has its own explicit position.

Core public loading now preserves original BF16 GR down/up/inject and head-up
matrices instead of requantizing all of them to Q8. Other native dense projections
keep the existing startup Q8 conversion. The full original Q6_K draft head is
also preserved and uses the existing native head consumer. Quotes derive from
these actual matrix formats and original page spans; there is no extra public
weight duplicate or converted model file.

The new front has20 private owners /4,685,824 admitted bytes. Core MTP context
is4096; the ordinary profile remains2048. Total core private plan is25,296,896
bytes inside a64 MiB reserve. Private pages, precharge, physical/full-region checks,
finite/padding guards, failure erasure and close receipt all include the new
owners. Cache/candidate data never leaves the owning request. BF16 GR uses the
same charged grouped buffer geometry as the target's native helper. Integer/Q8
planes receive exact padding checks rather than invalid F32 finiteness checks.

Connected callers:

- Prompt teaching submits all actual rows from each target carry chunk together,
  with real next tokens and per-row carries. One completed fault gate precedes
  the group's cursor publication; final prompt boundary stays prompt_len-1.
- Core accepted-prefix repair queues the original retained prefix as one front
  window after actual rewind. First confirmed-cell preservation and final cursor
  authority remain governed by AcceptedCacheRepair and target acceptance.
- Count1 full drafting/followup uses the same front with GPU-to-GPU copies of
  the real Token/Carry owners. It publishes actual residual/mixed/injection to
  the existing rest graph. Attention skips its old duplicate K/V append. No
  synthetic host token/carry is substituted for device followup authority.

Actual front dispatch accounting comes from its queued clears/projections/
operators/guards, not invented receipts. Failure clears all new front owners and
packed scratch before reuse. Ordinary feature-off math/call paths are retained.

Remaining fidelity gaps: cache owners still store FP16-rounded values in F32,
not native packed/paged half pools. The rest of MTP attention/expert/GR/commit
execution still contains legacy scalar families and must be ported together with
those packed consumers. Target prompt chunks remain three rows rather than native
MMQ prefill. Prompt groups still fence individually; stable front graph capture,
whole-prompt enqueue/final-fence scheduling and device-resident target carry
handoff remain open. Native1024-thread pipeline/hardware acceptance, numerical
KV/draft/repair comparisons and physical recovery have not been exercised. This
receipt is a connected front implementation, not complete MTP/engine parity.

Validation logs under `/media/jmicko/faststore/kuno-development`:

- `strata-mtp-front-build-20261014.log`: release core build passes.
- `strata-mtp-front-clippy-20261014.log`: strict release core Clippy passes.
- `strata-mtp-front-shaders-20261014.log`:216 offline WGSL/SPIR-V shaders pass,
  including the five new front kernels and reused GR/Q8/key norm consumers.
- `strata-mtp-front-geometry-tests-20261014.log`: two host tests pass: every
  MAX8 hidden-stream group/cache-tail interval and exact IQ4_XS lane layout
  versus the original independent canonical decoder. Not device parity.
- `strata-mtp-front-core-tests-20261014.log`: all nine accepted-prefix/source
  tests pass, including eight-row bounds. Old count3/private-byte assumptions
  in these affected tests now distinguish core versus ordinary contracts.
- `strata-mtp-front-ordinary-regression-20261014.log`:1,365 pass, one ignored.

Fmt/whitespace pass. Prior full-core failures remain open. No live GPU/model,
correctness/recovery or speed run, service/API/UI change or installation. Goal
active; full rewrite incomplete and speed gate closed. Next port the MTP rest
and packed cache as one consumer family, then continue prompt/PLE, graph/queues,
full sampling/input and final completeness/physical acceptance gates.

## Connected MTP packed cache and attention/GR rest (2026-10-14)

Source `294cb93`, isolated `codex/strata-rust-port`. Core cache append, dense
attention, rejected-tail rewind and full cache guards now share actual packed
FP16 K/V owners. Layout follows native `qsa_real_shapes` page size4:
`[page][two KV heads][four cells][256 half components]`. Each owner is4 MiB;
the page/cell inverse is used by rewind and guards, so an accepted prefix need
not occupy a contiguous byte prefix. The bounded single-sequence physical page
mapping is identity; a native allocated page-table owner is not yet ported.

Count1 draft attention consumes the front's prepared Q8 planes, original query
weights, native fused norm/RoPE, identity visible-cell selection,64-cell chunk
attention, max-rescaled merge, sigmoid gating and original output projection.
Partial/statistic owners cover all4096 visible cells. Fault flags are folded
before reuse. FF and final head reads use the same original BF16 native GR
family, actual preceding attention/MoE block and injection. Final residual carry
is published into the real existing followup owner. No synthetic carry or
host-generated replacement draft input is introduced.

The enlarged front/rest workspace has22 owners. Total MTP private plan is now
18,644,992 bytes, inside the existing64 MiB core reserve; ordinary plan remains
11,698,176 bytes. Packed buffers receive half-finiteness and real rejected-tail
checks rather than F32 scans. Allocation, erasure and close receipt use actual
owner counts. Original legacy scratch remains reserved and explicitly accounted.

Build, strict release core Clippy and221 offline WGSL/SPIR-V variants pass.
Three front/layout host tests and nine accepted-prefix/ownership tests pass;
feature-off direct-device regression passes1,366 tests with one ignored.
Logs: `strata-mtp-rest-{build,clippy,shaders,layout-tests,core-tests,ordinary-regression}-20261014.log`
under `/media/jmicko/faststore/kuno-development`. Formatting/whitespace pass.
These establish source/build consistency, not device numerics or model execution.
No GPU inference, speed comparison, service change or installation was performed.

Next coupled block: native MTP MoE. Read actual `record_rest`,
`s2_expert_grouped.dp.cpp`, `quantize_act.dp.cpp`, `native_bf16.dp.cpp`,
`native_router.dp.cpp`, `shared_expert.dp.cpp` and `native_moe.dp.cpp` together.
Existing Q2 kernels use floating-point activations; native S2 requires quantized
input and SwiGLU intermediate with separate F32 scales, matching integer
expansion/accumulation/reduction and real downstream consumption. Audit native
router and shared BF16 roles rather than keeping their startup Q8 conversion.
The native BF16 projection shown in `native_bf16.dp.cpp` uses ordered F32 FMAs;
do not infer F64 from the fallback router implementation or old chat summaries.

Full rewrite remains incomplete. Whole prompt MMQ/PLE, recorded graphs/queues,
arbitrary input/sampling, source/conversion audit, earlier full-core failures and
final numerical/state/recovery acceptance remain open. Speed gate stays closed.
