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
| GDN projection/conv/gates/recurrence | `verify.cpp`, `cuda/native_gdn*.dp.cpp`, `cuda/fused_gdn.dp.cpp` | GDN retained-state replay exists. Audit actual convolution/L2/alpha-beta/recurrence/output-norm fusion, grouped projections and retained commit inputs against pinned Intel source. |
| QSA projection/norm/RoPE/KV/indexer | `verify.cpp`, `cuda/native_qsa*.dp.cpp`, `cuda/qsa*.dp.cpp` | Causal Rust QSA exists. Audit fused norm/RoPE, KV format/append and block selection/gather, grouped consumers and tail repair; explicit F32 versus quantized KV difference needs resolution. |
| PLE source rows, projection and history | `verify.cpp`, `ngram`/`ple` kernels, PLE table reader | Original grouped reads/hash/history exist. Audit asynchronous deduplicated source reads, grouped BF16 projections and ordered history snapshots/commit. Never persist request n-grams. |
| Prompt prefill and streamed experts | `sycl/src/prefill/prefill.cpp`, `gemm.dp.cpp`, MoE MMQ/Intel implementations | Rust bounded graph prefill is not full prompt-path parity. Port connected chunk budget, streamed layer staging and dense/expert batches, including prompt history consumed by MTP. |
| MTP loading/prefill/draft/head | `src/core/mtp.cpp` as built by Intel target, Intel kernels | Original Q2 sidecar and public draft vocab exist. Audit native layout/conversion, draft graph/attention and prompt-state initialization; arbitrary text and full supported draft head must work. |
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
