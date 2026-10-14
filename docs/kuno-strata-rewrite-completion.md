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
| GPU route/group planning and CPU contribution | `verify.cpp` router, `resident_plan`, doorbell/pool/host plans; `src/kernels/cpu` | Native BF16 router/shared-gate projections and full512-expert softmax/winning-lane normalization are selected in the core. Physical/numerical acceptance and working Intel all-GPU-fetch versus host-pool mode/overlap still require audit. |
| Native expert math and grouped rows | `sycl/src/kernels/cuda/iq_kernels.dp.cpp::row_entries`, `native_gu_kernel`, `native_down_kernel` | Packed integer math/BDA roots exist. Separate GU roles, 4/2/1 chunks, activation, intermediate storage and downstream consumers are now selected together in the Intel core; all 158 WGSL/SPIR-V variants validate, including 105 physical-pointer checks; numerical/model execution acceptance remains pending. The global core now uses one per-role launch containing pinned row_entries4/2/1 loops, scalar IQ4_XS, native signed integer finish and XOR reduction; numerical/model execution acceptance remains pending. Historical fixed-chunk kernels are retained separately. |
| GR write/read/norm/down/up/injection | `sycl/src/kernels/cuda/fused_gr.dp.cpp::fused_gr_read_multi`; `verify.cpp::gr_read_group` | Native pending-write norm, sliced BF16 down/injection, fixed slice reduction and up/mix/commit are now connected and unconditionally selected in the core, including explicit PLE/final boundary writes. Resident original weights and the enlarged private scratch owner are charged together. Structural acceptance passes; numerical/model execution acceptance and final substitution audit remain pending. |
| GDN projection/conv/gates/recurrence | `verify.cpp`, `cuda/native_gdn*.dp.cpp`, `cuda/fused_gdn.dp.cpp`, `cuda/verify_kernels.dp.cpp`, `prefill/kernels.dp.cpp` | Core connects native BF16 alpha/beta, controls, recurrence/norm and accepted-prefix replay. Prompt scope selects contiguous H, causal convolution/history/L2, CB32 recurrence and directly consumed FP16 output. Source 9a7a23b connects prompt-capacity whole-chunk math without per-element status/replay owners, carved AB staging, and native row/value-head recurrent state shared by prompt/decode/repair. The large chunk caller/acceptance is connected in the latest controller receipt; remaining split verifier replay, physical execution and final numerical/queue audit remain open. |
| QSA projection/norm/RoPE/KV/indexer | `verify.cpp`, `cuda/native_qsa*.dp.cpp`, `cuda/qsa*.dp.cpp` | Core now connects FP16 KV for current and committed rows, native fused norm/RoPE, FP16-rounded raw index keys, retained pooled blocks, native Intel scores, 64-cell attention/merge and rejection clearing together. See the QSA receipt below for precise storage/scheduling substitutions and remaining numerical/state acceptance; do not claim full QSA fidelity yet. |
| PLE source rows, projection and history | `verify.cpp`, `ngram`/`ple` kernels, PLE table reader | Core connects native BF16 MAX8 projections, normalization/gate/conv/history, deduplicated positional reads and accepted-prefix ownership. Prompt calls now consume a bounded, joined double-buffered source prefetch. The latest controller receipt connects native large chunk geometry, joined uploads, retained tail and exact history acceptance. Native loans/queue scheduling and numerical/device acceptance remain open. Never persist request n-grams. |
| Prompt prefill and streamed experts | `sycl/src/prefill/prefill.cpp`, `gemm.dp.cpp`, Intel CMake build selection | Rounded BF16/FP16 GEMM, prompt GR/AB/head and original-source FP16 expert consumers connect to actual initial prompt calls. The bounded prompt path now uses logical-ID expert order, checked native transfer nodes, an eight-slot raw ring and two paired dequantization slots. The latest owned-consumer receipt connects full2048-row owners, non-overlapping raw/grouped route packets, causal math/state scopes, final-only head and full device MTP teaching (up to8192 matrix rows). Independent transfer/compute queues and real release boundaries are connected. Dynamic cache loans, conditional ring sizing and physical whole-model acceptance remain open. Pinned SYCL CMake does not build the CUDA/HIP MMQ/fused implementation. |
| Prompt dense/expert matrix backend | Intel oneMKL rounded BF16/FP16 GEMMs | Connected Rust cooperative8x16x16/F32 kernels now consume the actual rounded dense and grouped GU/DOWN images, with enabled device features, SPIR-V native registration and required16 subgroup in both pipeline paths. See the cooperative-matrix receipt; physical numerical/model acceptance remains open. |
| MTP loading/prefill/draft/head | `src/core/mtp.cpp` as built by Intel target, Intel kernels | Core connects MAX8 front, packed native FP16 cache, dense attention, BF16 FF/head GR and native scaled Q2/router/fused-shared consumers. Queue/branch/capture fidelity, numerical/state acceptance, arbitrary input and supported sampling remain required. See the front/rest/MoE receipts. |
| Suffix proposal and draft policy | Intel speculation implementation and `DraftPolicy` | Request-local Rust suffix/cost controller exists; audit supported window lengths and coupled sampling rather than fixing to four public-fixture rows. |
| Target verification and accepted-prefix commit | `sycl/src/core/verify.cpp` checkpoint/commit methods | Real greedy verification/state-only repair exists. Audit all active windows and acceptance cases against GDN/QSA/PLE/positions; independent correctness oracle remains missing. |
| Graph capture, queues and synchronization | `verify.cpp`, session/mtp graph construction, Intel queue/event implementation | Native Vulkan owned recording/replay now connects real compute/copy queues, concurrent buffer ownership, RAW/WAR/WAW timeline edges and joined completion. Actual descriptor/copy intervals and dequant-release signals are connected; physical ring overlap, stable complete branch recordings, scratch lifetimes and full execution acceptance remain open; a sequence of independent Wgpu launches is not equivalent. |
| Request lifecycle and arbitrary input | upstream CLI/session/inference entry points | Current Rust public-fixture gate is not a general engine. Connect bounded arbitrary prompts and request-local sampling/history, with no persistent content, protected memory and orderly erase/close. No gateway/UI work required. |

## Validation and the speed gate

This is a whole-engine port, not a search for individually faster changes. Keep
coupled upstream placement, arithmetic, consumers and scheduling together. A
component stays incomplete until its dependencies are implemented; incidental
timing cannot justify reverting it.

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

## Connected native MTP routed/shared MoE (2026-10-14)

Source `a4aa0dc`, isolated `codex/strata-rust-port`. Core `native_moe` is selected
unconditionally for the original Strata Q2 profile, replacing the old F32 expert
activation path as a complete producer/consumer family:

- Original BF16 router GEMV uses native256-thread ordered pair FMAs and XOR/block
  reduction. Native top10 computes full512-expert softmax, stable ID ties and
  selected-weight normalization accumulated by winning expert lane.
- S2 input quantization retains native reciprocal-multiply/half-away integer
  codes, separate F32 scales and signed chunk sums. Paired gate/up and paired
  down use native256-thread/eight-subgroup geometry, transposed activation/code
  words, exact integer dot-minus-sum, native ordered lane chunks and down-shuffle
  reduction. Real original top10 destinations are preserved; no expert remapping.
- Gate-major SwiGLU is quantized into the separate F32-scaled down activation
  owner. All512 experts remain resident in authenticated original four-shard
  roles. Native count1 routes are unique, so group/destination identity is actual
  rank; no host routing read or per-request weight transfer is introduced.
- Shared gate/up use one prepared dense Q8_1 input. Native fused SwiGLU/Q8
  produces the actual down planes; original BF16 shared scalar and native ordered
  routed/shared merge feed the existing real GR/head/carry consumers.

Router and shared scalar are preserved BF16 during loading. Original Q2 pages
remain read-only: their interleaved block headers/code fields are decoded directly
instead of allocating native S2's separate-code/scale conversion. Activation
transposition/scales/sums are kept in aligned owning planes instead of native
34-byte Q8 records; the paired integer products and scales are the same contract.
These layout substitutions still require physical numerical acceptance.

Three admitted private owners add131,072 bytes: input800 words, gate-major
gate/up12,800 F32 values and down2,000 words. Front/rest workspace now has25
owners and total MTP private plan18,776,064 bytes. Allocation, full padding
checks, erasure and close accounting include them. Integer planes do not receive
invalid F32 bit scans. Native fault/route/readiness proof remains connected.

Native dense Q8 preparation now admits the legal640-column shared projection,
without requiring Q6's256-value weight blocks. Its native quantizer and fused
shared producer use pinned `q8_1_finite.hpp`: half-away rounding and finite half
scale/sum clamping. Effective half scales/sums are present in the existing
private plane ABI; the unused second packed plane stays zero. All target/native
dense users share this corrected producer. Audit the original format-specific
dense minimum/correction consumers separately; this does not claim every dense
format has complete arithmetic parity.

Validation under `/media/jmicko/faststore/kuno-development`:

- Final `strata-mtp-moe-build-20261014.log` and strict release
  `strata-mtp-moe-clippy-20261014.log` pass.
- Final `strata-mtp-moe-shaders-20261014.log`:229 offline WGSL/SPIR-V variants
  pass, including all eight connected MoE variants and fused shared quantization.
- `strata-mtp-moe-math-tests-20261014.log`: independent canonical GGUF decoder
  agrees with S2 transposition/F32 scales/paired integer dots for both640 and2560
  inputs, original scale/code blocks, lane order and the all-zero case.
- `strata-mtp-moe-shared-tests-20261014.log`:640-column producer covers real1..8
  batched input/plane bounds; Q6 admission remains rejected for this geometry.
- `strata-mtp-moe-owner-tests-20261014.log`:nine accepted-prefix/owner tests pass
  with the new exact private plan.
- Feature-off regression at the pre-fusion checkpoint:
  `strata-mtp-moe-ordinary-regression-20261014.log`,1,367 pass/one ignored.
  Subsequent changes affect core shader bodies/bindings only; final core build,
  Clippy and offline shader checks cover those. No repeated feature-off suite.

Remaining MTP schedule gaps: shared/routed work is serialized on the owning
Vulkan queue rather than native side-queue fork/join; stable graph/capture and
whole-prompt enqueue/final-fence scheduling remain incomplete. GPU numerics,
MTP/target acceptance and physical recovery are untested. Earlier full-core
failures remain open. No live GPU/model or speed run, service/API/UI change,
installation or root action occurred. Full rewrite incomplete; goal active.

Next major coupled port is **Intel prompt prefill plus PLE/source history**.
Authoritative pinned `sycl/CMakeLists.txt:212` builds `gemm.dp.cpp`,
`kernels.dp.cpp`, `prefill.cpp` and `moe_fused_stub.cpp`, with sequential MKL
linkage. It does not define/build CUDA/HIP `STRATA_PREFILL_MMQ` or fused experts.
Port real Intel GEMM/chunk budget/streamed staging and its history consumers;
do not port the inactive MMQ stub or treat existing three-row verification
looping as prompt-path parity. Preserve the full graph/queue/input/sampling and
final completeness/state/recovery gates before any speed trial.

## Connected native PLE and target BF16/router math (2026-10-14)

Source `1018a09`, isolated `codex/strata-rust-port`. `strata_ple.rs`/WGSL
port actual `native_ple_postops.dp.cpp` and native BF16 multi-row arithmetic.
Core selection is unconditional for existing layer1 PLE consumers. Existing
19 private owners,6,258,688-byte MAIN plan, OLD/NEXT identity and whole-model
accepted-prefix commit remain authoritative; no new GPU content/weight root.

BF16 key/value projections read authentic original cache rows once across
real1..8 RHS, with native256-thread ordered pair FMAs and XOR/block reduction.
The same producer is connected to target router/shared-scalar BF16 roles.
Native global routing now computes full512-expert softmax before stable top10
selection, then normalizes by native winning-expert-lane accumulation rather
than the old top10-only exponential sum. Real remapping/grouping/usage consumers
remain unchanged; no substituted expert selection or per-request weight copy.

Native PLE tail connects1024-thread weighted RMS,512-thread materialized-product
signed-root gate,256-thread broadcast, native weighted normalization, ordered
dilated convolution/residual output and channel-major nine-cell history. Every
active status joins the original fault proof. The existing pending GR write is
flushed before PLE observation. Original accepted-prefix code consumes actual
normalized rows; enqueue does not commit host history or accept a candidate.

Layout/scheduling substitutions remain explicit: key/norm/query/result pages
stay distinct instead of native safe in-place aliases. Cached F32 convolution
weights are rounded to native FP16 at consumption instead of allocating another
public half copy. Causal history still uploads OLD and later reads actual
OLD/NORMALIZED/NEXT for the existing host-owned commit, unlike native mapped
device snapshots. Those history/queue changes still need the full connected
schedule audit and physical/numerical acceptance.

Core embedding reads use a wiping stable sorted/deduplicated plan. Original
token/head destinations are restored after worker completion. Four scoped
workers perform positional reads from cloned, already-inspected descriptors;
all join on success/cancellation/error before decoded scatter. Each worker's
96 KiB requested stack plus32 KiB runtime allowance and64 KiB raw/plan envelope
are included in the existing4 MiB host check. No row cache, row IDs, n-grams,
embeddings or prompts persist outside the request. `TensorView::positional_reader`
is a backend-only additive Rust API; it retains source ownership/stat checks and
does not reopen paths. This is bounded parallel I/O, not native io_uring or
whole-prompt next-chunk double-buffering; those remain required schedule work.

Validation logs in `/media/jmicko/faststore/kuno-development`:

- Final release build and strict core Clippy:
  `strata-ple-{build,clippy}-20261014.log` pass.
- `strata-ple-shaders-20261014.log`:235 offline WGSL/SPIR-V variants pass,
  including all six PLE/BF16 kernels and the updated global route.
- `strata-ple-math-tests-20261014.log`: three host tests pass. Native postops
  match the independent scalar F64 oracle through widths1..8; every history
  prefix has the exact original rolling-window values. BF16 pair order covers
  all2560 columns/all RHS. Full router normalization/stable ties match an
  independent full-softmax oracle. These do not establish GPU numerics.
- `strata-ple-owner-tests-20261014.log`: all12 source/host-bound/ownership
  tests pass, including deduplicated scatter, padding and history acceptance.
- `strata-ple-positional-tests-20261014.log`: actual parallel descriptor reads
  through the new tensor API preserve data and original seek cursor.
- `strata-ple-ordinary-regression-20261014.log`:1,370 pass/one ignored at the
  pre-router-extension checkpoint. Later changes select core-only BF16/router
  math and add its host oracle; final core build/shaders and focused tests pass.

Fmt/whitespace pass. No live GPU/model/speed test, root/service/reset/rebind,
API/protocol/UI change, installation or release. Required Vashti changes: none.
Earlier full-core failures and whole-engine acceptance remain open. Full rewrite
incomplete; goal active; speed gate closed.

Next: port **the actual Intel prompt GEMM/chunk/staging path**, including its
PLE next-chunk prefetch and target/MTP history consumers. Current global prompt
controller still calls three-row verification repeatedly. Increasing that loop's
count or wrapping it in a prefill name is not the native path. Read pinned
`sycl/src/prefill/{prefill.cpp,gemm.dp.cpp,kernels.dp.cpp}` and the real stage
allocation/ring/cache-loan contracts together. Preserve original public weights,
bounded private staging, final cursor/teacher alignment and failure erasure.

## Connected prompt operand/GEMM producers and consumers (2026-10-14)

Source `e64d770`, isolated `codex/strata-rust-port`. This is a genuine connected
prompt arithmetic block, **not complete prompt prefill**. The global controller
now enters a request-local prompt projection scope for initial prompt calls;
target verification/drafting still use their separate decode producers. Scope
ends after the actual completed chunk/fault/state boundary and before MTP teaching.
The initial controller remains a three-row loop: its larger native schedule is
still required, not inferred from this method's name.

`strata_prompt.rs`/WGSL use the already-admitted three64 MiB global stage roots,
with serialized reuse and no new GPU owners. Prompt matrix products pack BF16
activations for original BF16 matrices and FP16 activations/original decoded
weight images for packed dense matrices. Supported original dense formats are
Q8_0/Q4_K/Q5_K/Q6_K/IQ4_NL/IQ4_XS/BF16. Weight images are produced on GPU from
authenticated original cached rows; no converted file or whole F32 mirror.
The shared SwiGLU input follows native finite FP16 saturation. Tile counts and
all input/output intervals are checked against the real borrowed stages.

The Rust/Vulkan8x8x32 tile GEMM is the explicit oneMKL substitute: shared operand
tiles, F32 accumulation, original rounded operands and real output spans. It is
currently ordinary FMA computation, not a demonstrated Intel XMX/cooperative
matrix implementation. Backend reduction order/library tile schedule and
numerical/performance acceptance remain open; offline validity is not parity.

Connected consumers include normal, paired/bundled dense projections and final
head, prompt GR and GDN alpha/beta. GR now follows prompt256-thread residual
normalization, BF16 down/injection GEMMs, low-rank divide-by-four/SiLU/BF16 up,
ordered mixing and actual pending residual/injection writes. Original owners
remain charged, including the partial page now used for up output. GDN AB uses
BF16 operand products then real dt/A/sigmoid controls and the existing fault/
state consumers, without falling back to decode's F32-input native AB producer.
Panic/failure still uses the owning model's existing close/poison path; no prompt
or result is logged/persisted by the scope.

Explicit remaining substitutions: F32 normalized GR rows are retained rather
than native `gr_mix_r` recomputation. The three shared stage roots require serial
reuse; side-queue/shared overlap cannot be added without resolving their real
lifetimes/barriers. Original cache tile alignment/admission needs final audit,
especially the large decoded head tiles. Prompt attention/recurrence and routed
experts still use the current bounded verification families; streamed FP16 expert
GEMMs, ring ownership and budget/cache loans are not connected. Large prompt
chunks, PLE double-buffer prefetch, final prompt-only head selection, device
target/MTP carry teaching and whole-prompt fence scheduling remain required.

Validation logs under `/media/jmicko/faststore/kuno-development`:

- Final `strata-prompt-gemm-build-20261014.log` and strict release
  `strata-prompt-gemm-clippy-20261014.log` pass.
- `strata-prompt-gemm-shaders-20261014.log`:247 offline WGSL/SPIR-V variants
  pass, including twelve prompt producer/consumer variants and full head routing.
- `strata-prompt-gemm-math-tests-20261014.log`: two host checks pass. Original
  six-format weight images agree with independent canonical GGUF decoders;
  BF16/FP16 operand rounding and edge/partial tiles agree with independent F64
  products, covering1/3/8/17 RHS and1/7/8/19 output rows. These are not GPU or
  whole-model numerical checks. Later saturation/span/dummy-owner changes pass
  final build/Clippy/shader checks; no rerun of the unaffected host algorithms.

Fmt/whitespace pass. No GPU/model/speed run, root action, service/API/UI change,
installation or release. Earlier full-core failures and whole-engine acceptance
remain open. Full rewrite incomplete; goal active; speed gate closed. Required
Vashti changes: none. Continue native chunk/staging and expert GEMM ownership;
do not label the current small-window controller as completed prefill.

## Connected original-source prompt expert arithmetic (2026-10-14)

Runtime source commit `1cf7e50` on isolated `codex/strata-rust-port`.

`strata_prompt_experts.rs` / `.wgsl` now connect the actual initial-prompt scope
to native FP16 expert arithmetic. Reference: pinned Intel `prefill.cpp:3319-3370`
for logical-ID grouping and `3963-3984` for original dequantization, gate/up GEMM,
FP16-saturated SwiGLU and down GEMM. The prompt route flag sorts groups by logical
expert ID, retaining token/rank order within a group and the real physical remap.
Decode's routing, integer expert execution and CPU contribution stay separate.

Prompt execution queues every routed group on the GPU; it bypasses the decode
CPU/GPU split and its input/routing readback. `Plan` returns no remaining decode
GPU work or CPU jobs because prompt GPU work has already been queued; zero is
not a claim that the router selected zero experts. All original resident/HOST
roots are supplied by the actual `Sources`, including its publication and BDA
read dependencies. `fetch_role` moves the selected original HOST role cells to
stage2; resident cells are consumed directly. Stage0 holds gathered FP16 rows,
stage1 a single FP16 expert matrix. All three roots are reused in queue order.
No new private weight owner, whole-model duplicate or persistent conversion is
created. Original-format helpers/codebooks cover IQ2_S, IQ3_S, IQ3_XXS, IQ4_XS
gate/up and IQ4_NL/Q2_0 down. Native `hf_sat` infinity clamping and NaN rejection
are explicit in the SwiGLU consumer. Parent real rank weights, status/fault join,
reduction, accepted-prefix and cleanup owners remain authoritative.

The actual global stage owner allocates **three128 MiB roots**, not64 MiB. The
previous prompt receipt's64 MiB description and scope check were incorrect;
both prompt and enclosing stage scopes now accept the actual128 MiB owners.
Previous builds/offline checks did not prove this runtime ownership connection.
The historical64 MiB bank-fetch helper is separate from the selected global
path; do not confuse its constant with the global source staging allocation.

**Still incomplete:** the controller remains a three-row loop. This connected
arithmetic uses role-major serial staging across logical-ID groups, a separate
gate/up layout in the existing PART rows, FP16 values temporarily represented
in the existing F32 hidden owner, and portable8x8x32 FMA tile GEMM. Native uses
expert-major paired GU, staged copy/compute events and two dequant slots. Those
schedule/storage differences remain required work with the full chunk owner,
ring/cache loan, PLE prefetch, prompt causal families and target/MTP teaching.
Dispatch/INFO admission and stage lifetime/graph overlap must be audited together
with that owner. Offline validation is not device numerical/state acceptance,
XMX proof or throughput acceptance. Do not benchmark or discard this component
because its supporting schedule is unfinished.

Validation logs under `/media/jmicko/faststore/kuno-development`:

- `strata-prompt-experts-shaders-20261014.log`:256 offline WGSL/SPIR-V variants
  pass, including nine prompt expert variants and physical BDA checks for all
  six original expert formats.
- `strata-prompt-experts-clippy-20261014.log`: strict release core Clippy passes.
- `strata-prompt-experts-final-build-20261014.log`: final release core build passes.

No live GPU/model execution, numerical/state acceptance, speed test, root action,
service change, API/UI change, installation or release. Existing full-core unit
failures and whole-engine acceptance remain open. Full rewrite incomplete; goal
active; speed gate closed. Required Vashti changes: none.

## Connected prompt staging ring and paired expert order (2026-10-14)

Runtime source `9339904`, same isolated `codex/strata-rust-port`. This supersedes
the preceding role-major prompt expert schedule, without a timing experiment.
Pinned Intel `prefill.cpp:100` defines STAGE8, `:226` defines DQ2, and
`:3963-4012` connects native paired GU/dequantization to expert-major products.
The default non-stream-all branch is the reference for the current partial
resident cache; optional large stream-all rings and cache loans remain separate
work, rather than being falsely credited by an eight-slot declaration.

The real router map is read once through the owning admission (804 words). It
is validated for complete/distinct actual rank coverage using the existing
split validator with no CPU expert contribution. Its logical-ID order remains
unchanged; token activations are not read to the CPU. The caller queues up to
eight actual groups ahead, chooses raw slots modulo8 and dequant slots modulo2.
Every missing original role is a real HOST-to-private-stage buffer copy; resident
roles are read from their original BDA owners. Raw stage2 contains gate/up/down
bytes for each ring cell, including role offsets verified against actual graph
geometry. Two GU+D FP16 images borrow19,660,800 bytes of stage1. No new private
owner or second whole-model copy is allocated.

Gate/up dequantization now produces native interleaved1280-row FP16 matrices;
paired GEMM writes only the current group's interleaved F32 GU rows into the
existing256 KiB GU owner. Saturated SwiGLU writes packed FP16 hidden rows directly
into stage0, then DOWN consumes those rows and publishes the real route-indexed
PART/status/fault outputs. Earlier completed DOWN outputs cannot be overwritten
by later GU rows because those owners are distinct. The previous temporary
F32 representation of packed FP16 hidden rows is removed. Parent weighted rank
reduction, prefix acceptance and complete-owner cleanup remain connected.

**Checked transfer integration:** `PublicHostCopy` and
`ComputeClient::stage_public_host_copies` are additive internal Rust APIs in the
vendored CubeCL runtime. Only registered immutable HOST source bindings, complete
same-stream destination owners and checked aligned cell ranges are accepted.
The Wgpu backend resolves actual registry resources and validates physical
usage/extents and nonoverlapping destinations before queue mutation. Native
graph mode inserts real `vkCmdCopyBuffer` nodes in the same owned graph, with
retained source/destination resources and transfer/shader barriers. Other Wgpu
mode uses its owning encoder. There is no separate raw queue or untracked
submission; detach/close still joins the sole consuming stream. Original HOST
imports add only TRANSFER_SRC/COPY_SRC; external registration permits STORAGE
with optional COPY_SRC and retains readonly kernel lowering. COPY_DST, mapping
usage and mutable external destinations are not admitted by this copy API.

**Remaining fidelity gaps:** copies and compute still share the same owned
queue; this is not independent native copy/compute overlap. Dequant slots are
correctly ordered but not yet overlapped with later experts. The controller
still has three-row initial calls. Full chunk allocation/carving and byte
budget/cache loans, real larger grouped-row owners, prompt causal families,
PLE prefetch and device target/MTP teacher handoff remain required. Portable
FMA GEMM is still an unproven oneMKL/XMX substitute. Full graph/capture/admission
and whole-engine numerical/state/resource acceptance remain open. Do not run
speed trials or revert the component while its dependencies remain unfinished.

Validation logs under `/media/jmicko/faststore/kuno-development`:

- `strata-prompt-ring-final-build-20261014.log`: final release core build passes.
- `strata-prompt-ring-final-clippy-20261014.log`: strict release core Clippy passes
  for the operational implementation; later test-only fixture/comment changes
  pass final release build and the actual host test below.
- `strata-prompt-ring-shaders-20261014.log`:256 offline WGSL/SPIR-V variants pass,
  including paired physical-source GU and original DOWN formats.
- `strata-prompt-ring-copy-contract-20261014.log`: one host-only metadata contract
  test passes, rejecting private sources, readonly external destinations,
  foreign streams, sliced source views and out-of-owner/overflowing ranges. It
  uses an abstract registered metadata resource, not a GPU allocation or proof
  of physical copy execution. Run through the native package because vendored
  dependencies are not testable workspace members in this checkout.

Fmt/whitespace pass. No live GPU/model/copy execution, numerical/state/recovery
or speed test, root action, service/HTTP API/UI change, installation or release.
The earlier full-core failures remain open. Full rewrite incomplete; goal
active; speed gate closed. Required Vashti changes: none.

## Connected prompt GDN causal pipeline and FP16 consumer (2026-10-14)

Runtime source `810f89e`, isolated `codex/strata-rust-port`. Pinned reference:
`prefill.cpp:2855-2877` selects prompt projections, gates, conv, recurrence and
native output GEMM together; `prefill/kernels.dp.cpp:449-628` implements causal
conv/history/L2, `:876-991` implements the pipelined column recurrence, and
`:1643-1665` produces FP16 normalized/gated output. The Intel selector
`:1228-1234` disables the NVIDIA-specific key-head default; `:2634-2696` selects
CB32 pipelined columns and separate output norm. CUDA/HIP-only alternatives
were not substituted for that working Intel path.

`strata_prompt_gdn.rs` / `.wgsl` connect six operators to actual prompt calls:
64-token causal conv tiles, next raw history, token/head L2, split replay views,
128-thread CB32/four-row-group recurrence with next-token register prefetch,
and separate128-thread output norm. Existing ACT is the real contiguous
token-major H owner, RECURRENT holds raw readout, and the admitted activation
stage0 receives packed FP16 normalized/gated rows directly. Prompt projection
consumes that same image without a second F32 store/read/packing pass. There is
no saturating substitution: native output norm uses ordinary HF conversion;
nonfinite half outputs set the actual closing statuses and cannot publish.

The prepacked projection seam is internal Rust only. It authenticates the
actual scoped stage allocation/stream, exact contiguous input geometry and
output extents, rejects BF16-weight/F16-input mismatches, and retains original
weight tiles/consumers. Outputs cannot alias either activation or dequant stage.
Ordinary prompt products still pack their real F32 inputs; decode remains on
its separate fused recurrence/norm path. The reserved F32 GATED owner is unused
in prompt mode, rather than falsely credited as a normalized output producer.
No private root or whole-model duplicate is added. Existing status folds,
pending receipts, accepted-prefix state swaps and complete erasure remain owned
by the parent. New prompt kernel dispatches support bounded dynamic counts, but
the enclosing caller/receipts still have their existing MAX8 owners.

Explicit unresolved substitutions: state remains the existing **head-major**
layout, with equivalent logical accesses; native prompt state is row/value-head
major. Conv/recurrent next state is double buffered until whole-model acceptance
rather than written in place. Split Q/K/V replay inputs receive additional
writes from H for the existing repair owner. These are remaining placement and
state-contract audit work, not evidence of full prompt parity. Independent
queues, full chunk/carving/cache loans, causal prompt QSA, PLE prefetch, device
target/MTP teaching and final-only prompt head scheduling remain required. The
three-row controller is unchanged and is not completed native prefill.

Validation logs under `/media/jmicko/faststore/kuno-development`:

- `strata-prompt-gdn-final-build-20261014.log`: final release core build passes.
- `strata-prompt-gdn-clippy-20261014.log`: strict release core Clippy passes.
- `strata-prompt-gdn-shaders-20261014.log`:262 offline WGSL/SPIR-V variants pass,
  including all six connected prompt GDN operators and prepacked GEMM routing.
  Later host span/alias/parameter guards pass final build/Clippy; shader source
  and algorithms did not change after offline validation.

Fmt/whitespace pass. No live GPU/model execution, numerical/state/recovery or
speed test, root action, service/HTTP API/UI change, installation or release.
Earlier full-core failures and whole-engine acceptance remain open. Full
rewrite incomplete; goal active; speed gate closed. Required Vashti changes:
none. Keep all coupled prompt operators selected; finish remaining dependencies
before evaluating throughput.

## Connected prompt QSA consumers and PLE source prefetch (2026-10-14)

Source `242ce1c`, isolated `codex/strata-rust-port`. This continues the complete
port without opening the speed gate or changing the installed engine.

Prompt QSA now selects its real split, separate256-thread RMS and64-thread
RoPE operators for Q/K/index-Q/new pooled index blocks. Previously retained
blocks remain untouched. Rotation preserves the preceding normalization fault.
The final gate reads original interleaved Qfull and writes ordinary packed FP16
directly to the authenticated stage0 consumed by attn_output; it does not round
through the old F32 output owner. Actual closing flags/parent fault folds remain
connected. No new private GPU root or HTTP API is added. Kernels admit1..4096
rows, but the existing caller/owners still restrict the selected path to MAX8.

Pinned Intel selector correction: `qsa_prompt_attn.dp.cpp:2009-2010` calls the
XMX wrapper, then returns false to select older attention unless it succeeds.
`qsa_prompt_attn_xmx.dp.cpp:364-391` defaults STRATA_PROMPT_ATTN_XMX to zero;
XMX is built but opt-in. The current Rust prompt path retains its existing
FP16-KV/chunk attention/merge. This is not proof of native32-query scheduling
or of XMX arithmetic. Reference prompt producers/consumer are
`prefill/kernels.dp.cpp:1872-1950`, connected by `prefill.cpp:2880-3165`.

PLE source prefetch follows `prefill.cpp:2152-2184,2276-2351`: a scoped CPU
producer gathers upcoming original table rows while the current GPU chunk runs.
The rendezvous channel permits only current upload and one next embedding;
there is no detached worker or unbounded ready queue. Consumer checks binding,
start/count and every expected n-gram row before using the prepared image.
Rows/history/data wipe on drop; no prompt, row ID, embedding or result persists.
Receiver disconnect precedes worker join on normal/error exits; scope ownership
also joins on unwinding. Original positional-reader workers still all join.
An extra327,680 host bytes covers the additional131,072-byte embedding, bounded
coordinator stack and row/descriptor metadata; component and enclosing model
reservations both include it. No additional GPU embedding buffer is allocated.
The global teacher now indexes its already-validated next prompt token directly,
without the old helper's30/35-token fixture restriction. This alone does not
make the laboratory CLI an arbitrary-input engine.

Final source validation under `/media/jmicko/faststore/kuno-development`:
`strata-prompt-source-final-build-20261014.log` release build passes;
`strata-prompt-source-final-clippy-20261014.log` strict core Clippy passes;
`strata-prompt-source-final-shaders-20261014.log`266 offline shaders pass.
Fmt/whitespace pass. The final build log includes an earlier build's tail and
Cargo's artifact-lock wait; both commands exited0, and the subsequent Clippy
and offline shader invocations ran after the final build completed. No live GPU,
source-worker/model, numerical/state/recovery or speed execution occurred.
Earlier full-core failures remain unresolved. No service/API/UI/install/release
change; required Vashti changes: none.

**Next is the actual large prompt owner and schedule, not another MAX8 wrapper.**
Use native `carve`/`bytes_needed_impl` (`prefill.cpp:1127+,1832-1886`) for the
shared attention/MoE region and GR/carry lifetimes. Keep verification MAX8
separate. Connect larger route/member/status owners, layer-major full-chunk
consumers, actual budget/cache loans, device MTP teaching and final-only head.
Current prompt source prefetch is still bounded by the existing three-row loop;
native host upload events/double-buffer reuse remain part of that owner port.
For this artifact's large HOST complement, native `ring_slots:209-225` returns
STAGE8 by default; do not falsely require CUDA stream-all/fused branches.
Independent queues/capture, full storage/state audit, arbitrary-input/sampling
and whole-engine correctness/recovery acceptance still block speed testing.
Full rewrite incomplete; goal active.

## Connected chunk-capable prompt expert owners (2026-10-14)

Source `0f89b45`, same isolated branch. Prompt routing/grouping/products/reduction
now have their own1..256-row contract; verification remains MAX8. This is a
connected replacement used by current prompt calls, not a declaration that the
remaining prompt families or controller can already accept256 tokens.

The prompt router applies the existing full512 softmax/top10 algorithm to all
rows in subgroup batches, publishing actual logical/physical route pairs and
GPU probabilities. A bounded CPU grouping matches native Intel logical expert
order and stable token/rank membership. Its compressed8192-word map holds512
cells/lengths/offsets plus up to2560 rank indices. Checks cover route completeness,
per-token distinct experts, consistent logical/physical bijection and unique
members. No activations are read back. Decode keeps its original map and CPU/GPU
split. Prompt grouping still uses a checked readback/private upload rather than
native mapped pinned grouping storage: that queue/storage substitution is open.

Actual GU and rank outputs are carved from the existing third128 MiB stage:
GU `[64,66)` MiB, rank outputs `[66,98)` MiB. No additional GPU root is allocated.
Original raw expert copies are bounded below64 MiB, and raw compute sees only
that lower64 MiB view. Checked copies retain the full authoritative destination
root, as required by the existing copy contract. Stage1 retains paired FP16
weight images; stage0 retains gathered FP16 activations/hidden rows. GU/down
GEMMs launch all eight-row member tiles, including partial last tiles, with
local tile indices distinct from global member indices. Actual closing flags
and rank-weight reduction consume the carved outputs; the model acceptance
stamp and complete original-root cleanup remain connected. No stale MAX8 PART
page is consumed in prompt mode. Extra256 KiB host and8 MiB per-chunk I/O envelope
are included in the enclosing model plan.

Final release build, strict Clippy and268 offline shaders pass:
`strata-prompt-chunk-experts-final-build-20261014.log`,
`strata-prompt-chunk-experts-clippy-20261014.log`,
`strata-prompt-chunk-experts-shaders-20261014.log` under the development directory.
Fmt/whitespace pass. No live GPU/model execution, expert numerical or physical
view/copy acceptance, speed trial, service/API/UI change or deployment. Earlier
full-core failures remain open. Full rewrite incomplete; goal active.

Next: implement the real full prompt residual/GR/GDN/QSA/PLE/carry owner and
layer-major controller. Native `carve:1127+` overlays attention and MoE scratch
and preserves residual/GR/carry outside that region; `bytes_needed_impl:1832+`
counts the same layout. The existing three128 MiB roots provide real charged
storage to carve, but the current projection scope treats all of stage0/1 as
working images and rejects output aliases by allocation ID. Resolve their exact
working intervals and checked nonoverlapping views together before putting
persistent residuals or another phase into their upper halves. Preserve the new
stage2 raw/GU/PART intervals. Larger scratch/root cache loans still need the
actual budget policy; a256-row helper cap is not native budget parity.

Do not raise global MAX_BATCH or merely change step_by(3). Parent token packets,
NativeTargetRows, GR/GDN/QSA/PLE/shared owners, receipts and state commits still
use their small-window contracts. Prompt-only owners/acceptance, final-only head
and device MTP teaching must replace those dependencies together. Independent
queues/capture, dense/state-layout audit, general input/sampling and whole-engine
correctness/recovery remain required before speed testing.

Shared-shell sudo refresh now reports `a password is required`. Source work
remains possible without root. Do not start protected GPU acceptance or use a
privilege workaround; request a fresh credential when that work is ready.

## Connected carved GR workspace (2026-10-14)

Source `649e923`. Prompt storage now separates lower64 MiB GEMM/activation
images in stage0/1 from real carved upper-stage0 GR intermediates. Normalized
rows, down/low and up outputs use four disjoint256-aligned views with native
Alloc padding; their256-row capacities borrow the original charged root. The
existing prompt GR caller actually consumes these views. Its new handle entry
point supports1..256 rows without the legacy MAX8 intermediate geometry; parent
residual/injection/mixed owners and receipts are still the legacy small ones.
No new GPU root;4 KiB host metadata is charged. This is not full prompt ownership.

GEMM output admission now tests actual interval overlap rather than rejecting
all views of the same allocation. Prepacked activation admission checks exact
start/end as well as root/stream/extent, so an upper view cannot impersonate the
lower working image. Original scope construction still authenticates the full
128 MiB arena roots. Expert paired-weight shader guards now agree with the
actual lower64 MiB descriptor. Stage2 raw/GU/PART intervals remain unchanged.
Native Vulkan graph descriptors already preserve real offset/range; physical
view/copy and whole-model correctness are still untested.

Release build and268 offline shaders pass in `strata-prompt-gr-owner-{build,
shaders}-20261014.log`; final strict Clippy passes in the corresponding clippy
log. The only post-build change collapses an equivalent host guard for Clippy;
shader source/algorithms did not change. Fmt/whitespace pass. No GPU/model/state/
recovery/speed execution, root action, installation or API/UI change. Full-core
failures remain open. Goal active; sudo expired; no privilege workaround.

Next connect full prompt residual/attention/PLE/carry owners, prompt-only state
acceptance and final-only head/device MTP teaching to a true layer-major chunk
controller. Do not just enlarge the current three-row packets. Native GR still
retains F32 normalized rows instead of recomputing mix from residual/scales;
layout/queue/cache-loan/general-input audits and whole-engine acceptance remain
required before performance testing.

## Connected carved GDN prompt data and replay source (2026-10-14)

Source `81f1d76`. Actual prompt QKV/Z projections now write authenticated carved
views in stage1's upper half. Conv/history/L2 consume that raw QKV, recurrence
consumes the carved H and writes carved readout, and FP16 output norm reads the
actual carved readout/Z. Four256-row-capacity views occupy about32 MiB plus
native alignment/padding; lower64 MiB GEMM/paired expert images stay disjoint.
No GPU root is added. Existing state, Q/K/V replay splits, controls, status and
acceptance owners remain legacy MAX8: full GDN chunk ownership is not complete.

The new internal projector method accepts only exact registered carved outputs,
checks input/output overlap and retains actual cached tensor/client/fault
contracts. It is not an arbitrary writable handle API. Replay now snapshots the
real carved raw projection; its one native copy zeros inactive old-cache rows.
The new copy shader is included in offline validation. Extra73,728 runtime
metadata bytes is charged. Ordinary decode retains its original source/copy path.

Final release build, strict Clippy and269 offline shaders pass in
`strata-prompt-gdn-owner-final-build-20261014.log`,
`strata-prompt-gdn-owner-clippy-20261014.log`, and
`strata-prompt-gdn-owner-shaders-20261014.log`. A WGSL reserved identifier caught
by validation was renamed before the final successful build/shader checks.
Fmt/whitespace pass. No live GPU/copy/model/numerical/state/recovery/speed run,
root action, service/API/UI/install change. Earlier full-core failures remain
open. Full rewrite incomplete; goal active; last known sudo cache expired.

Next implement QSA's real shared prompt region and its coupled schedule. Native
`prefill.cpp:2950-3005` appends the chunk before scoring, uses shared completed
indexer blocks/dead-tail state, and selects queries in batches256; default
attention falls back to32-query batches (`:3124-3165`). Do NOT scale the old
per-query MEAN/POOL and LOGIT/PROB owners to256 rows: that duplicates shared
prefix state and hundreds of MiB of attention scratch. Native `qsa_set_bytes`
uses one bounded32-query attention workspace. Split only Q from interleaved
Qfull; gating already consumes Qfull directly, so a second full gate copy is
unnecessary. Preserve query-specific causal completed-block/tail visibility and
exact actual cache/position state when moving append earlier. Native KV paging,
indexer tail/dead layout and numerical fidelity still require the final audit.

Then finish full residual/PLE/carry ownership, prompt-only acceptance, final-only
head and device MTP teaching, plus native budget/cache loans and queue/capture.
The parent still runs three-row/MAX8 packets; do not label this as whole prefill
or alter global verification capacity to bypass remaining ownership work.

## Connected shared prompt QSA and native GPU selection (2026-10-14)

Source `6314ea4`, isolated `codex/strata-rust-port`. Prompt projections now
write the actual carved Qfull/K/V/index-Q/raw-index-K views. Split retains only
Q; gate consumes original interleaved Qfull. Q/K/index-Q normalization and RoPE,
packed candidates, shared means, attention output and bounded32-query partials/
statistics occupy14 disjoint upper-stage1 views, phase-overlaid with GDN. No new
scratch root. One newly completed index block is averaged, normalized and rotated
once for the whole chunk; query-specific score guards preserve causality. The
old per-query pooled copies are no longer prompt inputs. Exact scoped-output
admission and stream/interval ownership remain enforced. Metadata grows from
4KiB to16KiB, charged through the existing model quote.

Prompt append now precedes index scoring and attention. The attention loop
merges every tile before reusing its32-query scratch. Explicit query-row offsets
address full-chunk Q, selected IDs, output and closing flags; local tile indices
address partials/statistics. This avoids rebasing a2051-word selection stride to
an illegal descriptor offset. Attention reads actual appended authoritative KV.
The existing parent still provides only small windows;256-row geometry in these
helpers does not complete the parent controller or its receipts.

The source audit also corrected a pre-existing core selection mismatch. Pinned
`qsa.hpp::qsa_selection_width` is `min(visible,2051)`, including partial selection
of a block. The old whole-block selector could return only2048 cells at some
positions. Core attention, merge and closing-flag spans now use the native width.
The ordinary feature-off selector/geometry is unchanged and must not be treated
as the independent native oracle beyond that boundary.

Core QSA now unconditionally uses `block_topk_reg_kernel`'s connected GPU path:
1024 threads, register keys, per-warp radix histograms, four weighted8-bit passes,
subgroup/workgroup prefix scans, stable low-index ties and chronological cells.
The score producer uses native eight-warps-per-workgroup geometry, actual
normalized cell-zero dead key and tail bias. A separate ordered validation pass
joins finite/score-status failures into the existing sticky model fault before
selection. No native score readback, CPU sorting or selected-ID upload remains
in the model's `BatchSelector` path. Original non-core selection remains separate.

The dead key occupies an additional128 logical words after the completed-block
slots in each existing authoritative POOLED owner. Its initial mean preserves
the four explicit F32 additions, normalization and original rounded IK. This is
a storage substitution for native separate `idx_dead`; it is immutable after
cell zero. Stage-derived allocation quotes charge the real larger owners (at
4096 context, existing arena rounding adds6MiB over12 QSA layers). Shared prompt
mean storage does not duplicate this persistent state. The model I/O envelope
adds417,792 bytes for maximum-tile INFO and the selection/dead dispatches.
Core device admission now requires1024 invocations and32,912 shared bytes;
core batch contexts must be divisible by4 and remain bounded at4096.

Final release build, strict Clippy and all281 WGSL/SPIR-V variants pass:
`strata-qsa-coupled-validated-{build,clippy,shaders}-20261014.log` under
`/media/jmicko/faststore/kuno-development`. Reserved WGSL names found during
validation were corrected before these final runs. Fmt/whitespace pass. These
are offline checks: no GPU/model, physical view/copy, numerical selection/state,
repair/recovery or speed acceptance was performed. Earlier full-core failures
remain unresolved. No service, API, UI, installation or release change; canonical
checkout and inherited bytecode were preserved. Full rewrite incomplete; goal
active and speed gate closed. Last shared-shell sudo check required a password.

Next connect full prompt residual/mixed/block/carry owners and PLE producers to
the actual layer-major chunk controller. Keep GDN final-state acceptance separate
from MAX8 speculative replay; finish final-only head and device MTP teaching.
Do not raise global MAX_BATCH, enlarge per-query attention maps, or rename the
three-row loop as completion. Native cache-loan sizing, transfer/compute queues,
KV/indexer physical layout, remaining state/dense substitutions, arbitrary input,
sampling, and whole-engine numerical/state/recovery acceptance remain required.

## Connected full-chunk GDN math and native recurrent layout (2026-10-14)

Source `9a7a23b`, isolated `codex/strata-rust-port`. Actual prompt convolution,
L2, CB32 recurrence and FP16 output norm now call whole-chunk entries accepting
1..256 rows and join failures directly into the existing model fault. Their
arithmetic/state schedule is the same guarded source, not a different fallback.
Source substitutions assert every replaced closing binding/store exists so source
drift cannot silently retain an old status consumer. No per-element status or
split replay buffers are required by these full entries. The enclosing legacy
caller still supplies MAX8 splits for speculative replay, and two explicit
compatibility closing dispatches produce its old status views from the real
sticky fault. Those views cannot clear or overrule the original fault; enqueue
still grants no state acceptance. Full prompt acceptance/caller remains open.

AB's real raw BF16 projections moved out of the shared third128MiB root into
three carved stage1 views: raw AB, controls, and compact head-status capacity.
The existing caller now uses the raw AB view through `ab_handles`, which also
supports256 rows with caller-supplied full controls/status views. Existing
small control outputs/replay consumers are preserved. The three new views add
240KiB plus alignment inside the already admitted upper64MiB GDN region; they
add no GPU root or MAIN allocation. Actual output/input intervals and overflow
are checked. The retired full-root `control_stage` helper was removed. Two
compatibility INFO records per36 GDN stages add147,456 bytes to the global
runtime envelope and4,096 bytes to each component quote.

The native recurrent layout is now selected together for prompt, decode and
state-only accepted-prefix replay: `state[row][value_head][column]`. This matches
pinned `prefill/kernels.dp.cpp::gdn_rec_cols_kernel` and its pipelined variant,
`cuda/native_gdn.dp.cpp::step` and `cuda/fused_gdn.dp.cpp`. All connected native
input/output addresses use `((row * value_heads + head) *128 + column)`; no
head-major/native transpose bridge or additional state owner is introduced.
Snapshots, double-buffer swaps and finite-state inspections are opaque whole
owners and retain their existing bounds/lifetimes. Ordinary feature-off Cube
kernels and their layout are unchanged. This resolves the earlier source-layout
gap; physical/numerical/state transition acceptance is still unproven.

Final release build, strict Clippy, fmt/whitespace and286 offline WGSL/SPIR-V
variants pass in `strata-full-gdn-layout-final-{build,clippy,shaders}-20261014.log`
under `/media/jmicko/faststore/kuno-development`. A fast host-only check also
parsed/validated the12 generated GDN variants from the actual source factory;
it did not execute a device. No GPU/model/copy/numerical/state/recovery or speed
run, root action, service/API/UI/install/release change. Earlier full-core unit
failures remain unresolved; their old state-format assumptions need independent
review rather than being dismissed. Canonical checkout and bytecode preserved.
Full rewrite incomplete; goal active; speed gate closed; last sudo check expired.

Next finish full prompt QSA status/selection ownership and the residual/mixed/
block/carry/PLE controller as a connected path. QSA's32-query attention status
must also become a direct fault consumer: scaling the old24*2051-per-row status
would consume roughly50MiB at256 rows unnecessarily. Shared score/selection and
compact RMS-head flags can fit the remaining carved QSA region; mean flags for
all queries must not duplicate shared pooled state. GR and the new AB/GDN entry
points already accept real handles/counts256. Executor's original-weight prompt
projection and prepacked projection also accept handles/count through BatchSpans
without the legacy MAX8 validator. Keep old verifier receipts separate: they are
not large-prompt state acceptance. Complete final-only head/device MTP teaching,
cache loans, queue/capture/layout audit, general input/sampling and whole-engine
acceptance before any throughput comparison.

## Connected full prompt QSA ownership (2026-10-14)

Source `101f4e7`, isolated `codex/strata-rust-port`. The actual prompt caller
now enters `strata_qsa::prompt::run` after its five carved projections. That
connected1..256-row entry queues positions, Q-only split, compact Q/K/IQ RMS/
RoPE closing, FP16 candidate pack, dead initialization, authoritative append,
shared mean/norm/RoPE, native GPU weighted selection,32-query attention/merge
and the directly consumed packed gate output. All old per-operation prompt
math/selection branches are bypassed by this entry. It uses the stage's actual
cache/positions/pooled/fault owners; it does not authorize state acceptance.

Split, pack, append, attention, merge and gate join failures directly into the
existing sticky model fault. Norm uses only24/2/4 flags per query, and shared
pooled norms close only newly completed blocks; no per-query pooled copies or
24*2051 attention flags are needed. Score uses compact per-block flags plus a
direct fault for unsupported subgroup geometry. Norm explicitly writes failure
on that geometry too, preventing stale scoped flags from inheriting success.
Guarded source substitutions retain the original arithmetic and assert their
expected binding/store templates exist. Ordinary decode/MTP math is unchanged.

Six additional carved QSA views contain Q/K/IQ flags, scores, score flags and
selected cells. The whole20-view upper-stage1 region consumes59,490,304 bytes
including native alignment/padding, within the67,108,864-byte half. It remains
phase-overlaid with GDN. No GPU root or MAIN allocation is added. Scope metadata
reservation grows16KiB to32KiB, charged through the existing component/model
host quote. The conservative QSA INFO envelope now totals835,584 bytes, including
the existing tile records and17 full/compatibility records per12 layers.

The old MAX8 parent still receives12 explicit status adapters after the real
family, deriving its expected active masks from the actual model fault. The
old selector callback and later norm/pool/attention/etc events do not rerun the
family. Full prompt users need neither these adapters nor legacy scratch/replay
IDs. The enclosing controller/receipts are still three-row/MAX8; this is not a
claim of completed large-chunk scheduling or physical acceptance.

Release build, strict Clippy, fmt/whitespace and296 offline WGSL/SPIR-V variants
pass in `strata-full-qsa-path-{final-build,clippy,shaders}-20261014.log` under
`/media/jmicko/faststore/kuno-development`. A fast host-only parse/validation of
all generated QSA source factories also passed, and the region extent was
checked without a device. No live GPU/model/copy/numerical/state/recovery/speed
run, root action, API/UI/service/install/release change. Earlier full-core unit
failures remain open. Canonical checkout and inherited bytecode preserved.
Full rewrite incomplete; goal active and speed gate closed; last sudo expired.

Next compose the actual large controller and all its dependent consumers as one
source block, rather than another isolated kernel experiment. GR handles, AB,
GDN full math, QSA full math and expert grouping already accept256 rows. Native
outer R/mixed/block/carry ownership must persist outside attention/MoE overlays.
Refactor routed/shared/PLE callers to real handle/count entries, retain public
model binding and physical state-owner checks, and add prompt-only completed
state acceptance distinct from speculative MAX8 receipts. Extend the joined PLE
source's MAX8 row/data capacity with real charged host bounds. Finish final-only
head and direct device MTP teacher/carry input together with that controller.
Executor prompt and prepacked original-weight projection accept handles/counts;
do not reintroduce legacy validate_view/MAX_BATCH in those full paths. Native
byte-based loans, independent queues/capture, remaining substitutions, arbitrary
input/sampling and whole-engine correctness/recovery still precede speed trials.

## Connected large prompt controller and device MTP teaching (2026-10-14)

Source `f3196fc`, isolated `codex/strata-rust-port`.

The enclosing core prompt loop now uses real1..256-row chunks, rather than
calling the three-row verifier repeatedly. `resident_model_prompt.rs` connects
embedding, pending GR writes/reads, all48 GDN/QSA stages, routed/shared experts,
layer1 PLE, final residual materialization and a last-chunk/last-row-only head.
`strata_global_controller.rs` calls this path with a joined PLE source worker.
The ordinary MAX8 verifier and its speculative receipts remain separate.

The persistent region contains both R streams, mixed/attention/FFN blocks,
injection/router/shared intermediates, embedding and the retained PLE tail.
Together with GR it takes57,918,720 bytes of stage0's upper64MiB. PLE's phase
region takes57,695,232 bytes of stage1's upper64MiB and overlays the already
carved GDN/QSA regions; the working weight/activation images stay in the lower
halves. This adds no GPU root or MAIN allocation to the existing three128MiB
stages. The source envelope grows to12MiB, including its256KiB coordinator
stack and joined reader work. Another8MiB host and8MiB runtime reservation
covers controller uploads/maps/tickets/INFO; scope tables retain their32KiB
charge. All offsets are relative to the actual retained owning roots.

GDN/QSA have dedicated full-prompt enqueue/preflight/infallible-commit tickets,
with real binding, sequence, public-parameter identity, original state/cache
roots, start/count and shared poison checks. Whole-model fault completion,
PLE diagnostics, final output/carry if needed, a second fault proof, all-stage
preflight and global routed-stage stamp acceptance precede state publication.
A sealed completed-prompt certificate records the genuine boundary in the idle
MAX8 workspaces; it fabricates no queued MAX8 rows. Without that bridge the
existing post-prefill decode-policy/checkpoint checks would reject the model.
The request capability also has an explicit prefill-only256-row authorization;
it does not widen ordinary active/closed request or verification permissions.

PLE projects the full chunk, computes exact causal history into NEXT and retains
only the last `min(count,9)` normalized rows outside the phase overlay. The new
core `ple_transition::prepare_prompt` independently reconstructs NEXT from the
authoritative CPU history and that tail, verifies exact bits, and returns the
existing borrowing history ticket. This avoids a full10MiB activation map.
GPU/CPU histories publish together after all-family preflight. The original
scalar-history continuation oracle passes for1,8,9,32 and256 rows, including
signed zeros and abandoned proposals. The old MAX8 limit remains8.

The explicit GR boundary writer now consumes already activated prompt injection
once. Previously the decode writer applied its sigmoid a second time at a prompt
PLE/final boundary. Pending-write norm already consumed activated injection; the
new writer preserves that same convention through materialized boundaries.
The full shared path connects original gate/up/scalar/down projections, packed
saturating FP16 SiLU products, native ordered routed reduction and shared merge.

`resident_mtp_prompt.rs` implements the pinned Intel `Prefill::draft_kv` default
for the full chunk on the same admitted device. It borrows immutable target R1
and the now-idle phase region, normalizes embedding/hidden inputs, runs original
Q8 pages through FP16 prompt GEMMs, forms the conditioned residual, performs
BF16 GR followed by Q8 K/V projections and norm/RoPE, and appends native packed FP16 K/V. One
completed MTP fault/guard proof precedes its cursor update. No carry row maps or
per-eight-row teacher fences remain. FC-hidden treats each token's four streams
as contiguous rows, up to1024 rows. Only teacher token IDs are uploaded. Failure
retires the target and/or draft owners through their original terminal paths.
The model's last carry is mapped once for the existing decode controller.

Pinned-source audit clarifications:

- `prefill.cpp:225-253`: BF16X2 remainder GEMMs are opt-in; default is0.
  `gemm.dp.cpp::prompt_f16` returns false outside HIP. No missing default
  BF16X2 dependency is being silently replaced with a single plane.
- `sycl/CMakeLists.txt` does not select `STRATA_PREFILL_MMQ`; the fallback
  `mmq::built()` in `prefill.cpp:54-69` is false. Consequently the selected
  Intel teacher path uses FP16/Q8 GEMMs; decode's Q8 integer front alone was
  not the full prompt teacher implementation.
- `generate.cpp:2143-2172` defaults to owned prompt buffers at contexts<=32768
  and to borrowed cache space above that threshold unless flags override it.
  Do not assume cache lending is always the default of this4096-context path.
  The fixed256-row carve/reservation still needs the final native policy audit;
  it does not implement automatic2048-row sizing or arbitrary loans/refill.

This is a connected source implementation, not whole-engine numerical, state,
recovery or speed acceptance. Native queue/recording/transfer dependencies,
GEMM/layout substitutions, placement/loan selection, arbitrary input/sampling,
verifier repair and the unresolved full-core tests still require completion.
No physical GPU/model run, service/API/UI/install/release change occurred.
The canonical checkout and inherited bytecode were preserved. Goal active,
full rewrite incomplete, speed gate closed. Last shared-shell sudo refresh
reported a password required; source/offline work did not need root.

Validation for `f3196fc` (logs in `/media/jmicko/faststore/kuno-development`):

- `strata-large-controller-final-build-20261014.log`: release core build passes.
- `strata-large-controller-clippy-20261014.log`: strict release core Clippy passes.
- `strata-large-controller-shaders-20261014.log`:308 generated WGSL/SPIR-V
  variants pass, including physical-pointer validation. Host-only factory
  validation also checked all11 PLE variants and the large MTP norm variant.
- `strata-large-controller-host-tests-20261014.log`: all six core PLE history
  tests pass, including the independent long-chunk/scalar continuation test.
- `strata-large-controller-core-tests-20261014.log`: full core unit run is
  **not green**:1,149 passed,228 failed,1 ignored. An initial run exposed an
  outdated eight-row deduplicated-source limit assertion; it now tests the
  genuine256-row prompt contract while asserting verifier MAX_BATCH remains8.
  The final full run includes that correction and the new prefill-authority
  bounds/lifecycle assertions. No test is skipped to conceal these failures.

The200-failure historical log `strata-connected-gr-core-regression-20261014.log`
is from much earlier source, not the immediate parent.198 failures persist from
it;30 other tests fail now, and2 historical failures no longer fail. This does
not prove which intermediate change introduced each failure. Existing examples
include old four-row/host-profile fixtures, exact padding/INFO budgets and native
state geometry; independently audit each against its actual contract. Do not
call them all obsolete, claim no regression, or edit expectations wholesale.
The ordinary feature-off full suite was not rerun in this block.

Next connected source work before model/speed trials:

1. Complete the native queue/recording/transfer schedule and actual GEMM
   substitution audit. The current prompt transfer ring is still on one owning
   Wgpu stream. Native `prefill.cpp` uses a copy issuer and compute dependencies;
   merely having eight raw slots is not asynchronous queue parity. Actual Rust
   copy nodes enter `vendor/cubecl-wgpu/src/compute/stream.rs::stage_public_host_copies`
   and `strata_graph`; inspect their real barriers/lifetimes, not a fake event API.
2. The current8x8x32 prompt GEMM is scalar F32 arithmetic over the native rounded
   images, not a proved oneMKL matrix-instruction substitute. Useful existing
   Rust support: vendored `backend/vulkan.rs::register_cmma` recognizes FP16 and
   BF16 KHR cooperative matrices; `compiler/wgsl/compiler.rs` explicitly rejects
   `CoopMma`. The optional `cubecl-wgpu` `spirv` feature and existing
   `cubecl-core/src/runtime_tests/cmma.rs` show a Rust/SPIR-V route. Audit actual
   device capabilities and BF16/FP16 placement/data flow together; do not replace
   only one half or switch the path off because of an incidental timing.
3. Finish native owned/borrowed buffer policy and the configured chunk-size
   audit using the exact default branch above; native short-context default is
   owned buffers and2048 rows, whereas this bounded port currently uses256.
4. Connect arbitrary input/supported sampling, finish verification/repair and
   all228 test-failure audits, then perform the final whole-engine source
   completeness review. Only afterward run protected numerical/completion/state/
   recovery cases and then comparative throughput. Sudo is needed for protected
   content-bearing GPU tests, not ordinary source/build work.

### Matrix/queue inventory after the connected controller commit

Read-only Vulkan physical-device queries (no model, private workload or device
reset) succeeded. Public metadata logs in `kuno-development`:
`strata-matrix-capabilities-20261014.log` and
`strata-matrix-geometry-20261014.log`; the latter was produced by the same-named
standalone Rust/ash inventory source, not an inference engine.

The Intel BMG G31 driver advertises `VK_KHR_cooperative_matrix`,
`VK_KHR_shader_bfloat16` and `VK_NV_cooperative_matrix2`. Enumerated subgroup
matrices include8x16x16 FP16 operands with F32 accumulation and8x16x32 signed or
unsigned8-bit operands with32-bit accumulation. The older ash0.38 inventory also
prints component code1000141000 for another16-bit operand family, including
F32 accumulation; resolve that code against the modern Vulkan BF16 enumeration
before selecting it. This is capability evidence, not matrix execution proof.

Intel queue families0(graphics/compute/transfer),1(compute/transfer),2(transfer)
each expose one queue. The vendored Vulkan constructor currently requests only
family0's one queue (`backend/vulkan.rs:104-109`). Independent transfer work
therefore needs actual extra device queue creation plus semaphore/ownership and
tracked resource-lifetime integration; inventing a second family0 queue cannot
work. Existing graph copy nodes and their root ownership must remain valid.

The optional Rust SPIR-V feature is `cubecl/wgpu-spirv`, forwarding to
`cubecl-wgpu/spirv`; it is not enabled in this lab's current manifest. Its Vulkan
feature loader already knows cooperative matrices/BF16, but merely enabling the
feature does not prove the manually constructed target/MTP Wgpu device enabled
those extensions. `WgpuSetup` currently adopts an existing requested device.
Audit and connect device creation, kernel compiler selection, native rounded
operand owners and their consumers together. WGSL custom factories must keep
working alongside SPIR-V/Cube kernels. No runtime/default selection was changed
by this read-only inventory.


## Connected Rust cooperative prompt matrices (2026-10-14)

Source commit: `d326f23` on isolated `codex/strata-rust-port`.
The actual prompt/MTP dense projection and routed expert GU/DOWN consumers now
use Rust CubeCL cooperative SPIR-V kernels. This replaces the previously used
scalar8x8x32 WGSL GEMM; the old scalar sources remain offline historical coverage,
not a fallback selected when the matrix path is slow. The full rewrite remains
**incomplete**, and the speed gate stays closed.

### Connected compiler, device, owners and products

- `strata_matrix.rs` owns one8x16x16 tile per16-lane subgroup, native FP16 or BF16
  operands and F32 accumulation. Each masked shared load handles partial token
  rows without reading outside the original packed images. No host activation
  readback, C++ inference library, replacement weights or new global tensor root
  is introduced. Dense products retain arbitrary checked output strides; expert
  products retain the genuine grouping/member-to-route map, GU/activation/DOWN
  sequence, status acceptance and original fault word. Half-buffer arguments use
  two-byte element counts; treating these handles as u32 arrays would be wrong.
- `strata_prompt::project` calls the new dense kernel after its original pack and
  dequantization, including GR/GDN/QSA/PLE/shared projections and full device MTP
  teaching. `strata_prompt_experts::enqueue` calls the coupled FP16 GU and DOWN
  products against its existing dequantized ring images. Integer decode expert
  kernels remain the separately connected native math selected by the core.
- `strata-intel-core` now enables `cubecl/wgpu-spirv`. Both actual target and MTP
  device constructors call the native Intel device builder, enabling the matrix,
  BF16,16-bit storage and Vulkan memory-model features before adopting the device.
  The HAL receives the **actual full enabled extension list**, not the shorter
  original Wgpu-required list. Caller-admitted limits are preserved; the generic
  optional64-bit-indexing helper cannot silently increase storage-owner limits.
  A missing capability is an error, not a scalar fallback.
- Compiled SPIR-V is registered in the actual native graph with compiler-provided
  descriptor access/INFO layout and the cached kernel's real operator name.
  Read-only external imports still force their audited WGSL reflection/lowering;
  enabling SPIR-V does not bypass immutable host-model descriptor validation.
- Both Wgpu's ownership pipeline and the native recorded pipeline use the same
  geometry parser and explicit required16 subgroup. The unchanged physical
  integer shaders retain their32-lane requirement. Native graph construction
  checks support for both16 and32. Raw shader type alone is not sufficient to
  establish compatible subgroup geometry.

### SPIR-V legality and offline-only validation

Modern validation exposed Naga29's redundant ArrayStride on implicit Workgroup
arrays (VUID10684). One shared HAL helper now canonicalizes the words actually
passed to both Vulkan pipeline paths. It removes only implicit-workgroup-only
array strides, preserves explicit Workgroup Block layouts and host interfaces,
and rejects shared interface/implicit-workgroup type aliases instead of silently
rewriting a byte ABI. This changes decorations, not instructions, IDs or math.
The new host tests cover preservation, alias rejection, malformed streams and
subgroup geometry.

The distro2025 validator does not understand BF16 capability5118. The local
validation-only tool is built from official SPIRV-Tools
`fbe4f3ad913c44fe8700545f8ffe35d1382b7093` (vulkan-sdk-1.4.341.0, reports2026.1)
and its pinned SPIRV-Headers `04f10f650d514df88b76d25e83db360142c7b174`, under
`/media/jmicko/faststore/kuno-development/spirv-tools-matrix-20261014`.
This is not an installed inference dependency. Matrix validation compiles the
same four Cube expansions that are launched across all eight combinations of
standard uniform layout, uniform unsized arrays and explicit workgroup layout.
A configuration enables the validator's
`--uniform-buffer-standard-layout` exactly when that compiler/device feature is
selected; unsupported BF16 instructions are not skipped or downgraded.

Validation logs in `/media/jmicko/faststore/kuno-development`:

- `strata-matrix-connected-release-confirm-20261014.log`: release core build passes.
- `strata-matrix-connected-all-layouts-20261014.log`: all308 existing WGSL/SPIR-V
  variants pass, including122 physical-pointer validations. All32 cooperative
  variants pass modern Vulkan1.3 validation across the eight layout masks. The
  offline checker also verifies the actual cooperative multiply/fault opcodes
  and the same subgroup geometry used by both pipeline constructors.
- `strata-matrix-connected-clippy-20261014.log` and
  `strata-matrix-feature-off-clippy-20261014.log`: strict release Clippy receipts
  for the core and ordinary feature-off port, respectively.
- `strata-matrix-layout-tests-20261014.log`: all five focused host ABI/subgroup
  tests pass. No runtime/device/model is constructed by these tests.
- `git diff --check` passes. The inherited Python bytecode and canonical checkout
  were preserved. Initial validator-layout and Rust feature-owner borrow errors
  were corrected in the connected source; the final receipts supersede those
  earlier failed attempts.

No model content, GPU inference, numerical/state/recovery proof, timing trial,
service change, installation, release or Vashti/API change belongs to this block.
The earlier complete-core suite still has228 unaudited failures; focused shader
and ABI tests do not make that suite green.

### Required continuation: native copy/compute queues as one block

The constructor still requests only family0's one queue. An independent transfer
queue remains **unimplemented**. Inspection shows why adding a queue alone would
be incorrect: both `wgpu-hal::vulkan::Device::create_buffer` and the imported
`strata_global_host.rs` buffers use EXCLUSIVE sharing, while the same original
host experts can be read by shaders and copied into the raw staging ring.

Connect real queue creation/adoption, buffer sharing or paired ownership
transfers, copy/compute semaphore edges and retained recording lifetimes
**together**. The B70 exposes one queue each in families0/1/2, so family0 queue1
is invalid. Per-device queue metadata must describe queues actually created; do
not infer creation from the physical family list or a global environment flag.
If using concurrent sharing, apply it to both ordinary Wgpu roots and the actual
imported host copy sources before allocation. Compute-only imported buffers do
not need invented transfer usage. If using exclusive sharing, record both
release/acquire halves and real cross-queue dependencies.

Actual insertion/recording points:

- `stream.rs::stage_public_host_copies` validates/retains exact original sources
  and targets and currently inserts graph Copy nodes in stream order.
- `strata_graph.rs::{push_transfer,node_accesses,execute}` records genuine copies,
  fills, shaders and BDA read dependencies. Recording currently owns one compute
  pool/command/descriptor pool and roots until its completion fence; capture
  replay uses the exact resource/dispatch signature.
- `stream.rs::flush_inner` currently drains Wgpu parameter/upload writes before
  native execution. A new queue must not race that upload/erase boundary.
- Dependencies must cover RAW/WAR/WAW (including reused raw ring slots), indirect
  reads and hidden BDA source reads. Read/read need not serialize. Full-root
  access tracking is conservative; do not claim ring-slot overlap from it.
- Cached recordings must own both queues' command pools, synchronization objects
  and resources. Timeline values must advance on replay; an old signal value
  cannot be reused. The final fence must join both queues before info/source
  retirement, state acceptance, erase or recovery. A partial submission failure
  must retain all potentially in-flight owners; never free a copy pool merely
  because the compute fence was not submitted.

After that block, finish default2048/owned-versus-loaned prompt sizing and
placement audit, arbitrary request input/supported sampling, verifier repair and
all historical unit failures. Complete the source audit and protected
whole-engine numerical/completion/state/recovery acceptance before speed trials.


The B70 inventory's previously unresolved component value1000141000 is
`BFLOAT16_NV` in current Vulkan bindings (`tracel-ash0.39.5`). Its enumerated
8x16x16/F32 shape and advertised KHR BF16 features are the selected operand
capability. This resolves the public enumeration label, not device execution.
The final shared-shell `sudo -n -v` check still required a password; no protected
operation followed. Source/offline work was not blocked by that credential.


## Connected native compute/copy queue ownership (2026-10-14)

Source `ac05da8`, following `d326f23`. This is a connected scheduling
block, not hardware execution or a completed queue-fidelity claim. Speed gate
remains closed.

### Actual created queues and shared buffers

The Intel constructor now requests queue0 in a second transfer-capable family,
preferring the dedicated transfer family (the public B70 inventory's family2).
It does not request nonexistent family0 queue1. Before exposing the device it
adopts this actually created family into per-device HAL metadata, requiring
supported/enabled timeline semaphores. Generic feature-off device construction
keeps its original single queue. Missing independent capacity is an explicit
native-core setup error.

Ordinary Wgpu buffer creation on that device and the actual imported
`strata_global_host` copy sources both select CONCURRENT sharing for the exact
compute/copy families. Compute-only imported host descriptors retain their
existing usage; no extra mirror of expert bytes is created. The metadata is
stored with the owning device, not inferred later from physical capabilities,
an environment flag or a process-global registry. Textures/other devices retain
the original allocation path.

### Connected planner, recording, replay and completion

`strata_graph::execute` consumes `node_accesses` for the real retained shader,
BDA, indirect, copy and fill resources. `strata_schedule::Planner` streams one
node's accesses at a time, keeps both queues' latest readers/writers, and builds
actual command segments with backward RAW/WAR/WAW semaphore edges. Independent
buffers can overlap; read/read does not invent a data dependency. Every copy
queue acquires an initial compute-queue header after the real Wgpu
upload/initialization boundary. No host wait is inserted between segments.

Each `Recording` owns both command pools' buffers, its descriptor pool,
semaphores, retained resources and the complete plan. Same-queue barriers use
only that queue's supported pipeline stages; transfer-only command buffers do
not declare COMPUTE_SHADER stages. Two timeline semaphores carry real cross-queue
memory/execution dependencies. Replay reserves strictly new values before any
submission, including after a partial failed epoch. The final compute-queue
join waits for the last copy signal, and its real fence covers both queues
before state acceptance, source/INFO retirement, erase or normal owner release.

Readiness is published only after every command ends successfully. In-flight
state is set **before** the first queue submit, so a partial failure cannot free
the copy pool just because a final compute fence was never issued. Existing
terminal protected-lab teardown retains unknown in-flight owners rather than
resetting the GPU. Reentry cannot reset/reuse an unknown in-flight fence. This
is source-level lifetime handling; device-loss/recovery behavior still requires
protected execution and the final recovery audit. Diagnostic timestamp mode
serializes both queues/query reset and requires compatible timestamp bits; it is
not an uninstrumented speed result and stays disabled for interim model checks.

Native queue plans retain at most16 recordings with actual queue-vector metadata
capped at64MiB. Planner roots are capped at65,536 and nodes at the existing32,768;
there is no whole-graph duplicate of every access list. An additional80MiB host
reservation (retained cache plus bounded temporary planning) is charged through
the core prompt/controller lifetime quote. No global tensor/weight root is added.
Opaque driver command-pool allocations and real hardware memory admission remain
part of the final runtime/placement audit; this host reservation is not evidence
of measured Vulkan allocation usage.

Validation logs in `/media/jmicko/faststore/kuno-development`:

- `strata-dual-queue-build-20261014.log`: release core build passes.
- `strata-dual-queue-clippy-20261014.log` and
  `strata-dual-queue-feature-off-clippy-20261014.log`: strict core and ordinary
  feature-off release Clippy pass.
- `strata-dual-queue-host-tests-20261014.log`: six focused host tests pass. They
  cover upload acquisition/independent buffers, RAW/WAR/WAW and slot reuse,
  read/read, a pairwise oracle over2,048 operations, diagnostic ordering and
  monotonically reserved replay/partial epochs including overflow. No device or
  Vulkan queue is created by these tests.
- `strata-dual-queue-shaders-20261014.log`:308 existing shader variants,
  including122 physical-pointer validations, and32 cooperative variants pass
  the same modern offline validation as the preceding matrix block.
- `git diff --check` passes. Canonical checkout is still clean; inherited
  bytecode remains untracked/preserved. No live GPU/model, physical/numerical,
  state/recovery or speed acceptance was attempted.

These changes
do not alter the gateway, Vashti API/UI, installed service or model residency.
The earlier full-core228-failure result remains an unaudited historical baseline;
the complete unit suite has not been rerun for the changed host reservation.

### Important remaining queue fidelity gap: real ring-slot ranges

The planner currently keys hazards by whole VkBuffer, as the earlier recorded
backend did. That is correct/conservative, but the prompt's eight raw slots share
one64MiB buffer, so it prevents their intended transfer/compute overlap. Do not
call the independent-buffer host test proof of physical ring-slot overlap or
complete Strata queue parity.

Next connect byte-range hazards **and actual narrowed raw-slot descriptors** as
one block. `strata_prompt_experts::enqueue` already stages eight groups ahead in
`stage_prompt_group`, with exact raw slot offset/stride. Its Dequant currently
binds the whole stage2 lower64MiB; `strata_prompt_experts.wgsl` also requires
`arrayLength(raw)==16777216` and uses `p.r0` as that global slot offset. Simply
making the scheduler believe this descriptor is narrower would be unsound.
Change the real binding/view, shader offset/extent checks and planner together.
Preserve GU/up/DOWN source geometry, dequant slots, faults, private owner checks
and BDA read dependencies. In particular, the existing raw guard uses `p.r2`
even for GU; with an exact-sized slot, validate the actually accessed GU/up or
DOWN extent rather than assuming a gate blob is always smaller than DOWN.

The interval oracle must cover partially overlapping views, disjoint slots in
one buffer, opposite-queue readers/writers and later slot reuse. Keep interval
metadata bounded/charged, replay signatures sensitive to real views, and the
final join/lifetime guarantees above. Conservative barriers on the *same* queue
do not replace the required accurate cross-queue resource declarations.

After range/ring connection, finish whole-native graph/branch recording audit,
owned/default2048 versus loaned sizing and placement, arbitrary input/sampling,
verifier repair and all unit/source completeness checks. Only then protected
whole-engine correctness/completion/state/recovery and speed trials.


## Connected exact raw-slot ranges and dequant release (2026-10-14)

Source `3ca345c`, following queue `ac05da8`. The real prompt expert
path now binds one actual raw slot and releases it at the upstream dequant
boundary. This closes the prior whole-buffer/unused-stage dependency problem;
physical overlap, numerical correctness and full-engine acceptance are still
unproved. Speed gate stays closed.

### Declared ranges are real owned resources

`strata_graph::node_bindings` supplies both the existing same-queue barrier
accesses and the new cross-queue byte intervals. Shader descriptors use their
actual Wgpu offset/size, transfers their real source/destination spans, and BDA
and indirect reads retain their genuine resources. The scheduler does not infer
narrower accesses from kernel names, routes or private INFO values. Range changes
remain part of the exact existing capture/replay resource signature.

The interval planner splits at access boundaries, initializes uncovered gaps,
retains untouched prefixes/suffixes and rejoins equal neighboring states.
Opposite-queue RAW/WAR/WAW edges cover every overlapping interval. Disjoint views
within one VkBuffer can overlap; read/read has no new data edge. Same-queue
barriers may remain conservative without falsely inventing cross-queue read
ranges. Upload acquisition, monotone replay epochs, both queues' retained owners
and final joined completion from `ac05da8` remain intact.

Intervals are bounded at262,144, roots at65,536, and actual retained Vec capacities
at524,288 region cells. Contraction shrinks old oversized capacities, so repeating
fragmentation on different roots cannot hide unbounded retained allocations
behind a small live-region count. The core lifetime host charge increases from
80MiB to128MiB for native queue metadata/planning; the retained recording-vector
cap remains64MiB. This is a bounded host reservation, not measured driver memory
or hardware admission evidence.

### Native prompt raw image ABI

`strata_prompt_experts::raw_slot` creates an actual checked descriptor view of
one of the eight slots in stage2's lower64MiB. It rejects partial/wrong roots,
misaligned/unequal GU/up geometry and any ring entering the upper scratch half.
The existing group-ahead copy loop writes those same original role spans. No
extra weight root, host activation readback, conversion or weight mirror is added.

The shader uses slot-local `p.r0=0`, exact `p.r5` word length and separate GU/up
versus DOWN extent checks. This fixes the old guard's assumption that a gate
blob cannot exceed DOWN: the supported IQ4_XS/Q2_0 pair is a counterexample.
The field remains in the existing64-byte Params packet. The closed Down/GU image
variant also verifies its role. Original dequantization formulas, paired FP16
images, row/member mapping, faults and downstream matrix products are unchanged.
The operator cache identity carries `prompt-raw-slot-v2` for the changed interface.

Inspection found `Sources::bind` also attached all three fetch-stage roots as
possible BDA reads. Prompt image dequantization never uses their pointer table,
so retaining those declarations would still serialize its slot copies even with
correct explicit views. The actual prompt shader now has **no stage-address
binding**, and `bind_prompt_image` supplies only its original-model address table
and genuine per-layer GPU/HOST source owners. Those tables are built from
`gpu.roots()` and original HOST handles before stages attach, not scratch pointers.
The raw slot is tracked by its own actual descriptor. Ordinary decode consumers
keep the original stage table and stage read dependencies in `Sources::bind`.

### Release at the connected upstream event

Pinned Intel `prefill.cpp:3958-3983` dequantizes GU and DOWN, records `used[slot]`,
then submits the two GEMMs with SwiGLU between them. Releasing only at a later
queue change after GEMMs is conservative but loses the intended overlap.

The Down-image operator now declares a static `KernelMetadata::native_release_after`
boundary. Fresh and cached pipeline creation carry it into the native Program,
and the actual planner splits after that node even if the next node uses the
same compute queue. Its timeline signal can release the real raw slot to the
copy queue while GU/SwiGLU/DOWN continue from the separate dequantized owner.
There is no host wait, runtime tuning switch or request-derived marker.

The native release cache wrapper preserves original quant/format Info instead
of replacing it. Otherwise different DOWN quantizations could reuse the wrong
pipeline. External read-only masks still wrap that full identity. A false marker
preserves the ordinary cache key exactly. This is internal laboratory Rust
metadata; no Kuno/Vashti network API or installed package contract changes.

Validation logs in `/media/jmicko/faststore/kuno-development`:

- `strata-ring-ranges-build-20261014.log`: release core build passes.
- `strata-ring-ranges-clippy-20261014.log` and
  `strata-ring-ranges-feature-off-clippy-20261014.log`: strict core and ordinary
  feature-off release Clippy pass.
- `strata-ring-ranges-host-tests-20261014.log`: all11 queue tests pass, including
  two independent pairwise oracles over2,048 operations, mixed partial byte
  intervals/multiple bindings, preserved gaps/prefixes/suffixes, disjoint slots,
  real-reader reuse, dequant release before GEMMs and reserved-capacity contraction.
- `strata-ring-slot-tests-20261014.log`: three actual host view/cache tests pass.
  Original GU/up/DOWN spans are independently checked for all eight supported
  quant pairs and slots, including the gate-larger-than-DOWN counterexample;
  wrong/partial/misaligned/oversized owners are rejected; release keys preserve
  original quant identities and false preserves ordinary keys.
- `strata-ring-ranges-shaders-20261014.log`: all308 existing shader variants,
  including122 physical-pointer validations of the changed image interfaces,
  and32 cooperative variants pass modern offline validation.
- `git diff --check` passes; canonical checkout is clean and inherited bytecode
  remains untracked/preserved.

No live GPU,
model, numerical/state/recovery or speed run occurs in this block. The full-core
unit suite is not rerun; the earlier228 failures remain an unaudited baseline.
Canonical checkout, installed services/model residency and inherited bytecode
remain preserved.

### Next source work before whole-engine acceptance

Finish native prompt allocation/placement and size selection with **all dependent
owners and consumers**. The former capacity256/three128MiB layout was
not proof of native owned/default2048-row prefill or cache lending. Do not emulate
native2048-row grouping by eight separately routed256-row prompts and call it
parity. Native Intel defaults owned buffers at context<=32768, loans above that
unless flags decide otherwise; the actual selected policy must price the expert
cache and prompt/DQ/raw/PLE/MTP lifetimes together.

Do not assume the eight-slot ring is universally wrong either. Pinned
`prefill.cpp:100,212-226` uses STAGE8/DQ2, and even large chunks keep STAGE8 when
nonresident share is at least10% unless stream-all is explicitly forced. Its
larger byte-budget ring is conditional on stream-all eligibility and T>=1024.
Implement/audit the actual policy from the admitted model/cache data rather than
unconditionally reserving384/1024 slots. Retain the exact-slot and dequant-release
contracts while resizing/borrowing; do not disable them because of incidental
timings.

Then finish whole graph/branch recording, arbitrary input/supported sampling,
verifier/repair and the full source/unit-failure audits. Protected whole-engine
completion, numerical/state/recovery acceptance still precede any live speed
comparison. Driver allocation and physical queue overlap remain acceptance gaps.


## Connected native-owned prompt allocation foundation (2026-10-14)

Source `788e793`. **This is not completion of2048-row inference.** The
actual controller/grouping/math consumers still use `groups::CAPACITY=256`, and
cache lending is not implemented. Do not benchmark this intermediate assembly
or describe its buffer capacity as proof of native2048 grouping.

The allocator/ownership barrier for that port is now connected:

- `strata_prompt_allocation::Plan` records the native2048 default and selects
  Owned for context<=32768, CacheLoan above. Loan allocation deliberately returns
  an explicit incomplete-path error; it does not invent a lease or silently
  allocate unpriced permanent memory. Current engine context admission remains
  bounded below that threshold; general context/loan support is still required.
- The actual global model constructor allocates three separately tracked512MiB
  prompt roots in `PrivateArena`. `Workspace::requests` and `MAIN_BYTES` include
  all three before model/cache payload construction; their added MAIN charge is
  exactly1,610,612,736 bytes. Decode retains its distinct three128MiB fetch stages.
  These are conservative power-of-two Rust arena envelopes, not a claim of
  byte-identical SYCL allocation or physically measured VRAM use.
- `core_prompt_stages` supplies the real prompt Scope, while `core_stages` remains
  decode-specific. `Sources::attach_prompt_stage` attests the actual complete,
  same-stream raw owner through backend metadata. Original HOST copies must name
  that exact owner and remain below64MiB; scratch fetch address tables/decode
  dependencies still point to their original stages. Owned prompt refs participate
  in normal close and unknown-in-flight retention.
- Prompt working images remain exact64MiB views; persistent/phase views now carve
  from the larger owned roots. The same carving code is checked at both current256
  and planned2048 rows for GR/outer/PLE/GDN/QSA ranges and phase separation. Raw
  slot views, grouping scratch and protected erasure refer to the actual new
  owner, not a fictitious larger descriptor on an old allocation.
- Only the Intel-core build raises PrivateArena's maximum span/pool bins to512MiB;
  ordinary feature-off keeps128MiB. Native device admission requires both actual
  maximum buffer size and binding range to support the larger root before payload
  construction. Exact power-of-two exclusive pools still forbid a larger hidden
  owning tail.

### Bounded complete-root erasure inspection

The previous inspection read each complete owning buffer in one host lease,
which cannot scale to512MiB under the existing bounded I/O budget.
`OwnedZeroInspection` now reads exact ranges in32MiB windows. Every lease is
wiped/dropped and its staging sanitizer completed before the next range. A root
gets its single whole-buffer byte/count credit only after *every* window,
including padding/tail, passes. Errors/panics cannot publish a clean prefix or
release an uninspected root. GPU erasure itself still covers the full owned
word span using the real buffer length and2D flattened dispatch. Ordinary small
roots remain one inspection window.

This is connected to the real native cleanup readback backend, not a host-only
mock or an increased host allocation budget. The focused host tests additionally
exercise small inspection windows against retained synthetic pages, late tail
corruption and a last-window map failure. They do not prove live Vulkan erasure,
allocation, model completion or GPU recovery.

Validation logs in `/media/jmicko/faststore/kuno-development`:

- `strata-owned-prompt-build-20261014.log`: release core build passes.
- `strata-owned-prompt-clippy-20261014.log` and
  `strata-owned-prompt-feature-off-clippy-20261014.log`: strict core and ordinary
  feature-off release Clippy pass.
- `strata-owned-prompt-erasure-tests-20261014.log`:18 private-arena tests pass in
  the core build, including exact exclusive-page bins/large-span grid bounds,
  full-root window coverage and late-tail/map failure with bounded leases.
- `strata-owned-prompt-feature-off-tests-20261014.log`: the same18 tests pass
  with the original128MiB ordinary limit/pool bins. The deliberate synthetic
  panic messages are caught fault-injection cases; both results are green.
- `strata-owned-prompt-layout-tests-20261014.log` and
  `strata-owned-prompt-policy-tests-20261014.log`: actual carving code's256/2048
  persistent/phase bounds and policy/request/MAIN inventory checks pass.
- `strata-owned-prompt-slot-tests-20261014.log`: all three raw-view/cache tests
  pass against the new actual owner envelope.
- `strata-owned-prompt-shaders-20261014.log`: all308 existing shader variants
  (122 physical validations) and32 cooperative variants pass modern validation.
- `git diff --check` passes; canonical and inherited bytecode are preserved.

No live GPU/model,
numerical/state/recovery/speed execution, API/Vashti change, service change,
installation or release occurred. Full core unit suite is not rerun; the earlier
228-failure result remains an unaudited baseline, and changed root/byte fixtures
may require separate review. Do not rewrite those expectations wholesale.

### Required connected continuation: propagate true2048 consumers

Do this next, before loan support or any speed run:

1. Promote the actual prompt count/data path together: grouping, map/route/status
   layouts, expert GU/PART outputs, metadata/closing proofs, GR/GDN/QSA/PLE/MTP
   handle contracts, full controller chunks and source actor limits/charges.
   MAX_BATCH8 remains the speculative verifier limit; do not raise it to2048 or
   route eight separate256 chunks and call that native grouping. Shader
   `workgroup_size(256)` and CB32/attention32 tiling are different quantities from
   prompt row capacity and must not be blindly replaced.
2. The current MoE map's LOGICAL4100 intersects MEMBERS1540+2048*10, so changing
   count guards alone is wrong. Separate/price the raw logical-route packet and
   grouped map spans. Old map/status roots are decode-sized. The new prompt roots
   have room to carve actual prompt-specific map/status/GU/PART views without
   enlarging/reinterpreting decode's MAX8 owner. GU output needs16MiB and PART
   needs256MiB padded ownership at2048; both must feed their real matrix/reducer
   consumers with updated route stamps and closing acceptance.
3. `Storage::at_capacity(2048)` establishes only the listed persistent/phase view
   bounds, not every working-image or MTP consumer. Audit actual full R/teacher
   inputs and packed activation/weight images. MTP FC-hidden can expose four
   residual rows per token: native2048 teaching may need count8192 even though
   target prompt count is2048. Preserve separate logical geometries and real
   bounds; the current matrix limit4096 is not a universal prompt limit.
4. Derive PLE double-buffered source, map/readback, INFO and host/runtime charges
   from real row spans and joined actor lifetimes. Retain current last-nine-row
   history acceptance and completed-boundary publication, and verify new512MiB
   owners through bounded erasure rather than a single huge readback.
5. After full2048 connection, finish real cache loan/restoration/exclusion,
   conditional native ring/stream-all sizing and expert-cache placement against
   actual owned/borrowed bytes. The conservative512MiB envelopes are currently
   charged, but physical allocation/cache fit and exact native policy still need
   audit and acceptance. No fallback from an unimplemented loan is allowed.

Then complete arbitrary input/sampling, branch/recording and verifier repair,
source/unit-failure audits and protected whole-engine completion/numerical/state/
recovery gates. Only afterward compare throughput. Keep the full goal active.


## Connected native2048 owned prompt consumers (2026-10-14)

Source `d75549f`, following owned-root foundation `788e793`.
This supersedes the earlier256-row consumer limit for the native owned path.
**This is source/offline acceptance, not whole-model numerical or speed proof;
the complete rewrite and its benchmark gate remain unfinished.**

The actual controller, request authority and GR/GDN/QSA/PLE/MTP scopes now take
`groups::CAPACITY=2048`. The controller groups/routes the full chunk once per
layer; it does not split2048 into eight256-row controller runs. Speculative
verification, replay and ordinary decode still use their original MAX8 geometry.

### Actual MoE descriptors and closing proof

The third tracked512MiB prompt root now contains these real disjoint views
above its unchanged lower64MiB raw transfer ring:

| Role | Bytes | Purpose |
| --- | ---: | --- |
| GU | 16MiB | up to2048 group members by1280 F32 outputs |
| PART | 256MiB | up to2048 tokens by10 routed experts by2560 F32 outputs |
| raw route packet | 256KiB | physical routes at1540; logical routes at32768 |
| grouped map | 128KiB | ordered cell/length/offset/member packet |
| probabilities | 128KiB | all20480 active route weights |
| role status | 256KiB | GU and DOWN planes separated by20480 words |

`Storage` registers these actual views. The router writes the raw packet;
readback narrows that same view while preserving both existing byte offsets.
Host grouping creates the separate grouped map, and gather/dequant/matrix/
activation/reduction all bind these same prompt descriptors. The reducer no
longer accidentally rebinds decode-sized map/probability/status roots. Its
48-layer epoch/count acceptance remains joined to the real whole-model sticky
fault fence. Raw packet/probability/status views are reset per layer so a failed
producer cannot inherit proof flags from the previous layer. The prompt image
operator ABI key is now `prompt-owned-routes-v3`, preserving original quantized
identity and the existing native-release metadata wrapper.

At2048, MEMBERS1540+20480 ends before LOGICAL32768; the previous4100 split would
overlap. Grouped membership covers22020 words inside32768, probabilities cover
20480 inside32768, and both status planes cover40960 inside65536. Host route
readback/group/upload/validation lifetimes are charged at1MiB. Arena MAIN bytes
remain in the three already-priced512MiB roots, with no new hidden owner.

### Connected matrix, dispatch, state and source consumers

- Target GR/GDN/QSA and PLE consume full2048-row carved views. QSA whole-prompt
  norm/append/selection guards and the attention row-offset limit are raised
  together. Its attention scratch/tile remains32 rows and its per-layer INFO
  quote now includes all64 tile pairs; these are distinct geometries.
- MTP teaching uses up to8192 FC-hidden matrix rows for2048 target rows. Both
  actual projection and cooperative dense admission allow this, retaining real
  input/weight/output extent checks. Packed input remains in the lower64MiB
  working view; full residual and teacher normalization fit their actual80MiB
  carved views. No MAX8 teacher loop or per-row host activation carry is added.
- Copy, embedding broadcast, residual write and linear packing/dequant support
  flattened2D grids when their word spans exceed Vulkan's65535 X-workgroup
  limit. Their shader indexing uses the actual `num_workgroups`; non-linear
  GR/head/attention geometry is never flattened implicitly.
- PLE's completed-history transition accepts up to2048 while retaining only
  the independently verified last nine normalized rows and exact token history.
  MAX_BATCH8 acceptance is unchanged. The host test compares the tail with the
  scalar convolution/history owner across257/1024/2048 as well as smaller rows.
- PLE prefetch metadata is a wiping heap vector, avoiding a128KiB stack array.
  Its64MiB host quote covers current/prefetched20MiB embeddings, upload copy,
  original packed rows, plans and bounded threads. The outer model separately
  reserves64MiB host/runtime copies. These are conservative lifetime envelopes,
  not measured allocator/driver peak claims. The existing positional reader
  still reads each unique packed row once. Reconstruction now scatters by the
  stable destination map instead of scanning all destinations per unique row;
  repeated rows may be decoded again into the same wiping160-float scratch.

Validation logs in `/media/jmicko/faststore/kuno-development`:

- `strata-native-2048-final-build-20261014.log`: final core release binary builds.
- `strata-native-2048-clippy-20261014.log` and
  `strata-native-2048-feature-off-clippy-20261014.log`: strict core and ordinary
  feature-off release Clippy pass.
- `strata-native-2048-shaders-20261014.log`:308 shader variants pass, including
  122 physical validations and32 actual cooperative-matrix variants across all
  eight independent layout masks. This uses the modern validator recorded in
  the preceding matrix receipt; no GPU/device/model is constructed.
- `strata-native-2048-prompt-tests-20261014.log`: all12 focused prompt tests
  pass. These cover actual2048 carving/role spans, complete route roundtrip,
  tail corruption/duplicate rejection, non-overlap with the raw ring, flattened
  grid coverage, raw slot/cache identity and independent host arithmetic checks.
- `strata-native-2048-source-tests-20261014.log`: stable deduplicated source
  destination planning passes for every count1..=2048, retaining MAX_BATCH8.
- `strata-native-2048-ple-history-20261014.log`: the actual kuno-text history
  transition test passes against scalar convolution/history through2048 rows.
- `git diff --check` passes. Intermediate route-only logs are not a substitute
  for these final-source receipts; the first intermediate build's numeric type
  error was fixed before all final checks.

These focused host checks establish no actual GPU arithmetic, allocation,
recovery or whole-model state acceptance. The historical full-suite failures
are not waived by these green filters.

No live GPU/model/numerical/state/recovery/speed run, installed service/model
residency change, install, release, public API or Vashti change. Canonical
checkout and inherited bytecode are preserved. Full core unit suite is not
rerun; its earlier228-failure baseline still requires audit.

### Required continuation

Finish real cache loans/restoration/exclusion and the native conditional ring/
stream-all policy against actual cache placement/owner bytes. Current owned
roots remain conservative fixed512MiB envelopes; context>32768 lending is still
explicitly unimplemented, while current request/QSA/MTP context remains capped
at4096. Do not claim configurable native chunk/long-context parity from this
2048-row source change. Then finish arbitrary input/sampling, complete branch/
recording and verifier-repair audits, classify the historical unit failures and
run protected whole-engine completion/numerical/state/recovery acceptance.
Only after the documented completeness gate may throughput steer fixes.


## Cache contiguity and bounded public initialization (2026-10-14)

Source `16e99d8`, following native2048 consumers `d75549f`.
This is a connected model-loading prerequisite for lending, **not an implemented
cache loan, private borrowed owner, residency exclusion or restoration receipt**.

Pinned `sycl/src/program/generate.cpp:11524-11624` plans a tail loan, marks its
resident pairs unavailable before prompt initialization, and refills the actual
lent slots before publishing residency again. The queued refill waits before
its finite source ring wraps. Its default context<=32768 branch still owns
prompt buffers; the explicit/long-context borrowed branch remains required.
Kuno had packed the cache into16MiB GPU roots because the same constant also
controlled its host upload window. Those are different lifetimes and ownership
requirements: a bounded host upload does not require small GPU allocations.

The actual global plan now packs ranked expert blobs into roots of at most
512MiB, retaining original byte-aligned pair order and real per-role `Cell`
addresses. It still prices each exact physical root through `Cache::requests`,
`Sources::gpu_requests` and the existing persistent-bank admission. HOST
complements and original model bytes are unchanged; this does not manufacture
additional resident experts or a virtual contiguous buffer across allocations.
The final root may be smaller, and alignment/padding are actual charged bytes.

`Cache::load` allocates each real persistent GPU owner with `client.empty`, then
initializes *every byte*, including all padding, through bounded original-model
windows. `Plan::load_chunk_range` intersects the requested window with actual
gate/up/down source spans and reads the corresponding original tensor byte
ranges; everything outside those spans is explicitly zeroed. The upload view
preserves its full root identity and exact start/end offsets. Only immutable
model bytes use `write_public_owned_batch`; no private input is allowed here.
Each moved16MiB window drains before another is allocated, including its runtime
upload copy. The model source table/consumers are constructed only after all
roots have been initialized and attested as complete, distinct physical buffers.
The setup quote remains two16MiB windows plus the separate1MiB HOST reader;
it no longer follows GPU root size. The internal receipt reports root limit
and actual bounded host window separately. No GPU math/dispatch source changes.

Validation logs in `/media/jmicko/faststore/kuno-development`:

- `strata-contiguous-cache-build-20261014.log`: core release binary builds.
- `strata-contiguous-cache-clippy-20261014.log` and
  `strata-contiguous-cache-feature-off-clippy-20261014.log`: strict core and
  ordinary feature-off release Clippy pass.
- `strata-contiguous-cache-plan-tests-20261014.log`: all five plan tests pass.
  Existing tests preserve the independently recorded11570-pair ranked prefix
  and exact complete HOST complement at the same byte budget; empty/partial/
  full tiers and exact HOST reservation remain covered. New tests reassemble
  independently constructed original role bytes and all padding through1/7/
  31/257/1023-byte windows, reject empty/overflow/out-of-root ranges, and inspect
  real pinned-model512MiB root/16MiB window boundaries without a giant host Vec.
- `git diff --check` passes. GPU math sources are unchanged, so the prior308
  shader validations were not repeated for this model-loader change.

No live GPU allocation/model/numerical/state/recovery/speed acceptance, full
unit suite, installed service/residency change, install, release, public API or
Vashti change. Prior228-failure unit baseline remains unaudited. Canonical and
inherited bytecode preserved; no privileged work needed. Goal remains active.

Remaining: implement genuine cache-backed private ownership, exclusion and
original-format restoration in the connected prompt lifecycle. Do not pass a
model-public root to ordinary private upload/cleanup or invent512MiB authority
from a smaller cache tail. The borrowed owner must attest its actual complete
physical span, be included in content erasure/inspection, and remain retained on
unknown GPU completion. Borrowed pairs need an actual original-weight staging
source while their cache cells contain scratch; the existing HOST complement
contains only initially nonresident pairs. Per-layer address tables and retained
BDA dependencies must exclude leased cells, not just change a host bitmap.
On successful closure, restore original current physical placement/remaps (not
blindly initial profile bytes after adaptive swaps) and publish only after a
joined completion. Then finish conditional ring sizing/stream-all and remaining
source/completeness/numerical/state/recovery gates. No speed benchmark yet.


## Exact physical-parent prompt view geometry (2026-10-14)

Source `c0e9efc`, following contiguous-cache loading `16e99d8`.
**This removes an actual consumer-side lending blocker; it does not create or
register borrowed private roots. Cache loans/restoration remain incomplete.**

The real private Scope, Storage carving, MoE control/GU/PART views, raw transfer
slot slicer and source-stage attachment now use each complete parent's actual
byte size. A model-cache root may be alignment-sized and slightly below512MiB;
using the old constant in `offset_end` would either reject it or claim bytes it
does not own. Structural parent checks reject slices, mismatched whole/used
sizes, non256-byte alignment and sizes outside64..512MiB. Storage additionally
rejects duplicate allocation identities and foreign streams. The actual family
carve determines whether each full parent is large enough; there is no promise
that every64MiB root can contain the2048-row phases.

The lower working slices remain exactly64MiB, all existing matrix/route/status
view lengths stay unchanged, and raw transfers remain in the lower64MiB. Source
attachment proves the real full buffer offset/size through retained backend
metadata and requires that its actual MoE views fit. Copy destinations must
match the attached root's exact size and allocation identity. No fictitious
512MiB view is made from a smaller root; no math shader, copy/ring policy,
allocation quote or default three512MiB owned allocation changes.

`whole_root_bytes` is a structural envelope check, **not a public-to-private
conversion capability, physical ownership proof, MAIN credit or erasure proof**.
The running Scope still obtains BufferIds from the ordinary private arena;
its exclusive power-of-two allocator and full owned-root cleanup are unchanged.
A future loan must register actual cache backing and its full erasure under
an exclusive source-lending transaction before private Scope publication.

Validation logs in `/media/jmicko/faststore/kuno-development`:

- `strata-exact-prompt-parent-build-20261014.log`: core release build passes.
- `strata-exact-prompt-parent-clippy-20261014.log` and
  `strata-exact-prompt-parent-feature-off-clippy-20261014.log`: strict core and
  ordinary feature-off release Clippy pass.
- `strata-exact-prompt-parent-tests-20261014.log`: all15 focused prompt tests
  pass. No GPU math source changes, so prior shader validation was not repeated.
- `git diff --check` passes; canonical and inherited bytecode are preserved.

The three new tests check independent non-power-of-two parents, derive their
exact family minimum sizes from actual view ends (including native alignment
slack), accept those minima and reject a256-byte shortage in each parent.
They also reject aliases/slices/foreign streams/oversized roots and prove raw
slot offsets use the true owning end. These are host layout tests, not live
Vulkan allocation or borrowed ownership/erasure acceptance.

Next implement the real private borrowed-root registration and source-exclusion/
staging/restore transaction described in the preceding cache receipt. Only
then is it meaningful to connect configured loan selection and conditional
ring/stream-all policy. Complete the broader source/unit/whole-model acceptance
before allowing throughput to steer changes. Do not replace missing ownership
with a host bitmap, fake Handle or unpriced fallback allocation.


## Registered private cache roots and erasure-before-restoration transaction (2026-10-14)

Source `09351ea` plus audit-purpose separation `03ea215`, following actual-parent consumers `c0e9efc`.
**The source-owned borrowing component exists; the prompt controller does not
select it yet. Leased-pair prompt staging/table dependencies and allocation/
configuration integration remain incomplete. This is not complete loan-mode
inference or a live GPU/erasure/restoration acceptance receipt.**

### Real private registration without a new allocation

`PrivateArena::register_cache_roots_on_configured` registers three complete
cache buffers under an exclusive source-owner transaction. It allocates no GPU
root, performs no public/private upload substitution and does not reconfigure
I/O. Each actual full backing must match retained backend offset/size/buffer
size, be distinct and same-stream, and have bounded256-byte-aligned physical
size. All handles are retained before the first zero initialization. The local
ledger measures exact private exposure of already-precharged persistent banks;
it is not fresh MAIN allocation credit or a bank-release receipt.

A private `borrowed_complete` layout admits those exact physical sizes without
rounding a model root up to the next power of two. Ordinary `Ledger::reserve`
still requires the original power-of-two allocator layout; the separate private
registration route uses the exact borrowed layout. A registered arena cannot
allocate more roots or reset/reuse its IDs. Its final cleanup always runs the
real complete-root erasers and32MiB bounded readback inspection, including the
short last window, and verifies all owned bytes/root counts. Failed cleanup
retires the shared context and retains unaudited handles. The old public-fixture
retained-erasure shortcut is rejected for borrowed roots. Mandatory private
cleanup uses `strict_zero_audit`; borrowing never sets the public-fixture audit
flag or grants its replay permission. The audited receipt and failed-handle
quarantine recognize strict private coverage independently of public purpose.

### Source exclusion, current placement and restoration

`Sources::with_private_cache_roots` takes the actual cache's three root indices,
validates full prompt carving against them, joins previous readers, and sets
loan/mutation exclusion before registration can erase weights. The callback
receives only the private arena and its IDs. All ordinary prototype/bind/HOST
borrow/ready paths remain revoked; the exchange publication path cannot clear a
loan. The source now owns the actual executor-adopted retirement Arc, rather
than trusting a new per-call flag. Its unsafe closure contract requires one
serialized protected worker and no escaping ID, handle, Scope or scheduled replay.

On body success, error or panic, private close/inspection runs first. The actual
`finish_borrowed_roots` gate accepts only the exact complete three-root audit;
errors, panics, byte-prefix or root-count mismatches cannot enter restoration.
Restoration writes every byte/padding from original model artifacts through
16MiB windows into the same persistent roots, draining each upload. Its errors
or panics cannot reopen sources. Unknown registration/cleanup/restore completion
retires the real context, leaves source exclusion in place and quarantines the
owning roots. Only completed erasure AND restoration clear both exclusions;
a body error/panic is still returned after safe restoration. Closed arena Drop
is idempotent and cannot erase newly restored model weights.

Restoring the initial profile would be wrong after adaptive exchanges. A wiping
physical-to-logical `Placement` now accompanies the actual source transaction.
Exchange proposals authenticate incoming/outgoing identities in current cells
and prepare its next inverse map before hardware writes; `finish_publication`
publishes it only after real role exchange, private remap/usage completion and
adaptive-owner commit. Failed/partial publication remains blocked. The restorer
uses current logical IDs with the original per-layer role byte layout. Both
current and pending49152-byte maps plus metadata are explicitly charged through
an additional128KiB HOST reservation; neither is logged or persisted.

Validation logs in `/media/jmicko/faststore/kuno-development`:

- `strata-private-cache-loan-purpose-build-20261014.log`: final core release builds.
- `strata-private-cache-loan-purpose-clippy-20261014.log` and
  `strata-private-cache-loan-purpose-off-clippy-20261014.log`: final strict core
  and ordinary feature-off release Clippy pass.
- `strata-private-cache-loan-purpose-core-tests-20261014.log`: all20 arena tests
  pass. New cases require exact non-power-of-two coverage while ordinary owned
  bins still reject it, and prove all three short final windows are inspected
  before credit; corrupt or failed last-window maps cannot publish a prefix.
- `strata-private-cache-loan-purpose-off-tests-20261014.log`: the same20 arena
  tests pass with the ordinary128MiB allocator cap and original private-allocation
  policy. Borrowed registration remains disabled in that feature-off build.
- `strata-private-cache-loan-completion-tests-20261014.log`: both source tests
  pass. The actual completion gate is fault-injected at erasure/restore errors
  and panics, missing-root and byte-prefix receipts; restoration never starts
  without complete erasure. The inverse placement retains all logical IDs and
  correct cells after repeated swaps and rejects stale identities.
- `strata-private-cache-loan-plan-tests-20261014.log`: all five original source-
  plan/HOST reservation/window reconstruction checks pass with the extra128KiB
  metadata charge. No arithmetic shader source changes, so prior shader variants
  were not repeated for these owner/control changes.

Synthetic panic messages are caught fault injections and the test results are
green. These host checks do not prove actual Vulkan registration, model-byte
restoration, erasure, state, recovery or whole-engine correctness. Full unit
suite is not rerun; historical228 failures remain unaudited.

### Required connected continuation

1. Connect a prompt-only leased-source capability that can stage lent resident
   pairs from actual original weights, patch physical source flags/tables and
   omit scratch roots from BDA read dependencies. The HOST complement has only
   initially nonresident pairs. Do not simply relax mutation guards: current
   borrowing deliberately blocks every model source consumer and cannot run MoE.
2. Thread the registered arena's real parent IDs through Scope and the full
   prompt controller; support lending as a configuration/allocation choice.
   Existing `Plan::CacheLoan` allocation is still explicitly unimplemented,
   default model construction still allocates three512MiB owned prompt roots,
   and request/QSA/MTP context is still capped4096. Do not pretend that keeping
   unused owned buffers constitutes faithful loan pricing/cache placement.
3. Retain persistent charge exactly once while transferring private coverage;
   finish native conditional ring/stream-all and auto chunk sizing together.
4. Validate actual mapped GGUF restoration after adaptive swaps by comparing
   the physical roots before/after protected private exposure, including body
   failure, cancellation, tail/map failure and unknown GPU completion. The host
   gate tests prove ordering/policy, not actual Vulkan cleanup or recovery.
5. Finish branch/recording, arbitrary input/sampling, verifier repair and source/
   historical-unit-failure audits, then whole-engine numerical/state/recovery
   acceptance. Only after the full completeness gate may throughput steer work.

No installed services/residency, public API/Vashti, installation or release
changes. Canonical and inherited bytecode are preserved; no privileged/GPU work
occurred in this source block. The goal remains active and incomplete.
