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

**Selected Intel mode, corrected 2026-10-15:** the actual tested SYCL serving
launcher sets `STRATA_VERIFY_DEVICE_PLAN=1` and `STRATA_VERIFY_NO_HOST=1`.
The saved comparison receipt confirms those settings. Its GPU groups both VRAM
experts and the mapped HOST complement into the primary pointer plan and reads
either source directly. It skips the per-layer CPU expert service/doorbells.
For the B70/Xe port, reproduce this mode with complete original source ownership;
CPU expert splitting/doorbell servicing and `--spec-split` A/B windows are optional
upstream modes, not missing prerequisites of this selected path. Earlier notes
below requiring those modes before completion are superseded on that point.
Whole-window graph/parameter reuse, target/MTP state, prompt behavior and correct
source placement remain required. Do not replace missing experts with other IDs
or remove the system-RAM source complement.

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
| Exact dense placement and expert budget | `sycl/src/program/generate.cpp` model/cache setup; `sycl/src/core/expert_cache.cpp::open_sized` | Exact dense roots exist. Native global byte-prefix planning and an owning bounded GPU loader now compile; graph validation consumes the real artifact plan. Experimental core construction now invokes the sole global owner before prefill, and the core48-layer coordinator selects global routing/HOST fetch/role execution. Physical/numerical acceptance remains unfinished. All777 required original float tensors already match the selected native format (484 BF16,292 F32,1 F16); generic conversion is not required for this artifact. Target embedding currently occupies GPU memory while the measured source uses mapped HOST; this residency substitution remains explicit. The ordinary feature-off engine retains uniform banks. |
| Low-RAM host complement and promotion | `generate.cpp` source/adaptive tier; `sycl/src/core/verify.cpp` fetch/rebase | Original host complement, direct exchange and GPU selected fetch exist. Audit full byte ownership and asynchronous safe publication against upstream; do not duplicate the whole model in RAM. |
| GPU route/group planning and original HOST access | `verify.cpp` / `verify_kernels.dp.cpp::resident_plan_kernel`, selected SYCL `DEVICE_PLAN=1,NO_HOST=1` | Native BF16 router and full512-expert normalization exist. Correct selected path reads actual VRAM or mapped HOST aliases directly on the GPU, preserving every selected expert. Source36d13d3 replaces the CPU split/staging consumer with direct authentic source aliases and removes per-layer CPU joins/submissions; protected execution is pending. Generic CPU doorbells/split windows are optional modes, not this B70 completion gate. Physical/numerical acceptance remains required. |
| Native expert math and grouped rows | `sycl/src/kernels/cuda/iq_kernels.dp.cpp::row_entries`, `native_gu_kernel`, `native_down_kernel` | Packed integer math/BDA roots exist. Separate GU roles, 4/2/1 chunks, activation, intermediate storage and downstream consumers are now selected together in the Intel core; all 158 WGSL/SPIR-V variants validate, including 105 physical-pointer checks; numerical/model execution acceptance remains pending. The global core now uses one per-role launch containing pinned row_entries4/2/1 loops, scalar IQ4_XS, native signed integer finish and XOR reduction; numerical/model execution acceptance remains pending. Historical fixed-chunk kernels are retained separately. |
| GR write/read/norm/down/up/injection | `sycl/src/kernels/cuda/fused_gr.dp.cpp::fused_gr_read_multi`; `verify.cpp::gr_read_group` | Native pending-write norm, sliced BF16 down/injection, fixed slice reduction and up/mix/commit are now connected and unconditionally selected in the core, including explicit PLE/final boundary writes. Resident original weights and the enlarged private scratch owner are charged together. Structural acceptance passes; numerical/model execution acceptance and final substitution audit remain pending. |
| GDN projection/conv/gates/recurrence | `verify.cpp`, `cuda/native_gdn*.dp.cpp`, `cuda/fused_gdn.dp.cpp`, `cuda/verify_kernels.dp.cpp`, `prefill/kernels.dp.cpp` | Core connects native BF16 alpha/beta, controls, recurrence/norm and accepted-prefix replay. Prompt scope selects contiguous H, causal convolution/history/L2, CB32 recurrence and directly consumed FP16 output. Source 9a7a23b connects prompt-capacity whole-chunk math without per-element status/replay owners, carved AB staging, and native row/value-head recurrent state shared by prompt/decode/repair. The large chunk caller/acceptance is connected in the latest controller receipt; physical execution and final numerical/queue audit remain open. Split verifier mode is an optional upstream path, not selected by the measured server. |
| QSA projection/norm/RoPE/KV/indexer | `verify.cpp`, `cuda/native_qsa*.dp.cpp`, `cuda/qsa*.dp.cpp` | Core now connects INT8 KV for current and committed rows, native fused norm/RoPE, FP16-rounded raw index keys, retained pooled blocks, native Intel scores, 64-cell attention/merge and rejection clearing together. The measured server explicitly selects INT8 for target and MTP; source2a3d1ad ports both together and the combined CPU shader/layout checks now pass. Numerical/GPU acceptance is pending. Earlier generic-default FP16 scope claims below are superseded. Numerical/state acceptance remains required. |
| PLE source rows, projection and history | `verify.cpp`, `ngram`/`ple` kernels, PLE table reader | Core connects native BF16 MAX8 projections, normalization/gate/conv/history, deduplicated positional reads and accepted-prefix ownership. Prompt calls now consume a bounded, joined double-buffered source prefetch. The latest controller receipt connects native large chunk geometry, joined uploads, retained tail and exact history acceptance. Sources `595393e` and `57147a8` make actual decode and prompt history GPU-authoritative in existing OLD/NEXT roots, with checked CPU token-only publication, GPU accepted-prefix history and independent PLE capture roles. Default prompt/decode float readbacks and prompt diagnostic tail copies are removed; explicit diagnostics and the legacy CPU oracle remain. Source `131d29c` selects the native scoped accepted-prefix consumer before state/cursor publication; the diagnostic checkpoint path keeps the older schedule. Pending event/graph fidelity and numerical/device acceptance remain open. Prompt loans are disabled in the measured32768-context configuration and are not a completion prerequisite. Never persist request n-grams. |
| Prompt prefill and streamed experts | `sycl/src/prefill/prefill.cpp`, `gemm.dp.cpp`, Intel CMake build selection | Rounded BF16/FP16 GEMM, prompt GR/AB/head and original-source FP16 expert consumers connect to actual initial prompt calls. The bounded prompt path now uses logical-ID expert order, checked native transfer nodes, an eight-slot raw ring and two paired dequantization slots. The latest owned-consumer receipt connects full2048-row owners, non-overlapping raw/grouped route packets, causal math/state scopes, the unbatched final prompt input uses normal count1 decode for the first head, and full device MTP teaching covers the preceding prefix (up to8192 matrix rows). Independent transfer/compute queues and real release boundaries are connected. The measured server uses4096 prompt rows and32768 context, versus current2048/4096. Its no-borrow selection and nonresident fraction select the existing eight-slot/two-dequant-ring path; alternative loans/ring sizes are not prerequisites. Physical whole-model acceptance remains open. Pinned SYCL CMake does not build the CUDA/HIP MMQ/fused implementation. |
| Prompt dense/expert matrix backend | Intel oneMKL rounded BF16/FP16 GEMMs | Connected Rust cooperative8x16x16/F32 kernels now consume the actual rounded dense and grouped GU/DOWN images, with enabled device features, SPIR-V native registration and required16 subgroup in both pipeline paths. See the cooperative-matrix receipt; physical numerical/model acceptance remains open. |
| MTP loading/prefill/draft/head | `src/core/mtp.cpp` as built by Intel target, Intel kernels | Core connects MAX8 front, packed native INT8 cache, dense attention, BF16 FF/head GR and native scaled Q2/router/fused-shared consumers. Positive-threshold chains now keep hidden carries on device and observe the same token/probability/fault packet as zero-threshold chains; mapped/event scheduling remains open. First/grouped device-source and ids-only target result consumers now exist; the shared-context constructor receipt below supersedes the separate-client blocker. Core construction now selects the actual target client and one combined account. Source9de7fef pins the original target embedding/head once, shares them with MTP and gathers target embeddings on GPU. Actual device/state acceptance is pending. Source0bfbea8 connects the original-order CJK106299-row Q6 draft head, inverse-ID history mapping, compact confidence and coupled sampling; source2a3d1ad connects INT8 caches. Combined CPU checks pass, with full numerical/GPU execution still pending. Source `862f5fa` separates first-front staging and actual observation from capture, and prestages all coupled draft controls once per round. Source `8262367` connects typed MTP phases/4096 boundaries and original model-instance graph namespaces; source `d0670b0` connects direct original MTP replay and capture-derived host/sampler handback. Physical acceptance, commit/event fidelity, numerical/state acceptance and runtime policy remain required. See the latest device-end receipt. |
| Suffix proposal and draft policy | Intel speculation implementation and `DraftPolicy` | Sourcea26adc2 ports the native bounded trigram index, longest/recent suffix choice, cost priors, probing/stale rules and optional lookup-chain accounting into the selected controller. Configured MTP/target windows no longer shrink at the output cap; state commit and emission are separate. Coupled sampling and physical state acceptance still need end-to-end validation. |
| Target verification and accepted-prefix commit | `sycl/src/core/verify.cpp` checkpoint/commit methods | Source `131d29c` connects real provisional rows to accepted GDN/QSA/PLE candidates while original full-window transactions remain open, then publishes only n after real fault/I/O proof. Count1 uses its candidate and all-accept multirow runs original state replay. Old diagnostic full-window repair is not selected by the native controller. Source `4083258` connects the retained multirow commit graph with explicit dynamic accepted-count/position fields, original family preflight and count1 skip. Source `91e1f9e` connects original pending commit export/join; `1f46f76` connects a scoped original next-draft consumer before explicit commit acknowledgement. Source `0bf0b5f` emits real verified predictions before next drafting, matching the source delivery order. Global waits still serialize GPU execution. Independent draft/readiness/INFO dependencies, original storage substitutions and independent GPU/numerical oracle remain open. |
| Graph capture, queues and synchronization | `verify.cpp`, session/mtp graph construction, Intel queue/event implementation | Native Vulkan owned recording/replay now connects real compute/copy queues, concurrent buffer ownership, RAW/WAR/WAW timeline edges and joined completion. Actual descriptor/copy intervals and dequant-release signals are connected; Source `d40442c` connects direct retained target submission with typed control uploads and original family receipts; source `d0670b0` connects direct MTP invocation with original actual packet observations and sampler/actor handback. Source `b6732c8` adopts independent MTP compute and concurrent buffer families with semaphore handoffs. Source `414e2c3` connects actual current INFO copies on the native transfer timeline and avoids empty primary submissions; source `86f4c35` replaces the global single fence/last-recording assumption with actual lane fences and stable original recording flights. Source `131d29c` connects synchronous original accepted-prefix publication; source `4083258` connects the retained commit graph. Source `91e1f9e` connects original pending submission identities and exact joins; `1f46f76` connects scoped original MTP consumption before explicit commit acknowledgement. Real independent readiness/INFO/draft execution remain open; source `543bc73` couples original pending commits to retained shared MAIN debt; source `a8e538f` retains actual recording ranges and preflights whole INFO-copy conflicts; source `d2d3228` pins original verified R, submits its real readiness dependency before commit and scopes all unkeyed/named draft lane selection. Source `7ad47ca` connects original native packet fill/copy, real HOST_READ availability, final fence and checked HAL host map without primary Wgpu readback copies/callbacks. Source `7cb1cbc` connects native upload state/render/flush/unmap and all-owner joined I/O reuse without Wgpu remaps or primary reuse clears. Global drains and selective original-owner reuse/admission still require the coupled dependency change. Global drain removal and draft/event fidelity remain incomplete. Physical ring overlap, scratch lifetimes and full execution acceptance remain open. Measured serving lazily captures branches; eager all-branch warmup is not a prerequisite; a sequence of independent Wgpu launches is not equivalent. |
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


## Connected prompt-only cache sources and one-session controller (2026-10-14)

Source `60f6760` + aggregate-GR correction `12b310f` + model rebase `8b503e2`, following `f82d00c`, isolated `codex/strata-rust-port`.
This connects the previously missing original-weight fallback and prompt/source
transaction. **It is not whole-port completion, long-context acceptance or a
speed result. The compiled public fixture remains4096 tokens with owned prompt
buffers. Its explicit loan HOST grant is zero.**

### Original model sources and memory accounting

- `strata_prompt_shadow_plan.rs` maps every role of every cell in the three real
  lent roots to bounded original-weight HOST pages. Pages are at most128MiB,
  starts256-byte aligned, with actual driver-backing envelopes rounded to1MiB.
  A page contains bytes from exactly one authenticated GGUF shard; mixed-layer
  and mixed-role roots do not falsely inherit the first shard's identity. At
  most30 pages and4MiB extra setup/metadata are quoted. No unleased cell is
  invented, and the exact plan instance is checked.
- `strata_prompt_shadow.rs` allocates/maps/initializes/imports/registers these
  actual non-device-local HOST_VISIBLE buffers through the same owning Vulkan
  code as the original complement. Full allocation/padding is initialized;
  bounded1MiB wiping source reads select the current inverse placement, including
  prior adaptive exchanges. The three actual cache roots are retained, not
  replaced with fake descriptors. Setup holds a separately explicit RAM grant.
- The persistent fallback is refreshed under joined source exclusion on each
  loan; it does not detach/re-register every request or exhaust lifetime slots.
  Existing authority/owner-before-exposure, joined detach, erase and retention
  on unknown exposure apply to both complement and fallback pages.
- The original immutable-source contract now has an explicit checked model-only
  rebase transaction (`8b503e2`). A non-cloneable token atomically revokes
  whole distinct source roots before the backend flushes/joins prior readers.
  Warm fallback refresh and actual adaptive GPU/HOST swaps use this authority;
  only fully initialized/flushed original model bytes and completed raw writes
  can resume readers. A dropped/failed token leaves them revoked; it cannot be
  replaced by ordinary uploads or a forced active flag. Token metadata is quoted
  against the registry's live bound, with identities/counters preserved. Private
  data is still forbidden in every registered source.
- The typed CubeCL external registry now supports joined monotonic budget
  extension on its owning Vulkan stream. It preserves live/retired allocation
  identities, cumulative counters and non-recycled slots. A replacement vector
  is admitted against its old+new metadata peak before any state change. Shrink,
  closing entries, foreign/unhealthy streams and insufficient grants reject.

### Actual consumers and transaction

`Sources::begin_prompt_loan` joins/excludes prior readers, refreshes original
weights and republishes the actual physical-cell pointer tables before private
registration clears any lent byte. Every leased role points at its real HOST
fallback with HOST flag1. BDA read dependencies exclude the lent GPU roots;
prototype buffers also resolve to original weights. Prompt HOST copies keep the
registered `stage_public_host_copies` path and native transfer queue/acquisition/
header/ring lifetime; no ordinary untracked upload replaces a transfer node.

The new owning `BorrowedPrompt` token supplies its actual registered arena IDs
to Scope. The request controller holds one session/Scope across all prompt
chunks and device MTP teaching. Prompt-only bind/staging/scheduler boundaries
accept that state; ordinary decode, CPU expert borrowing and adaptive mutation
remain excluded. Ordinary model close refuses an active loan and retains owners.
Abandonment retires the actual shared context and attempts private cleanup.

After Scope drops, finish requires complete private erase/readback inspection,
then restores every actual cache byte from current logical placement. Only then
are normal model tables/read sets and previous prompt-stage attachment restored
and joined. Failure/panic cannot publish residency. The closed private arena is
dropped before normal publication and cannot erase restored original weights.

Owned-versus-loan selection now flows through actual Workspace allocation and
MAIN request pricing: loan mode omits all three512MiB owned buffers. Constructor
shape validation compares the equivalent native shape while admission charges
only the actual request. The local model arena now uses the exact already held
MAIN grant rather than its obsolete2GiB ceiling, which could not contain the
new default owned prompt buffers plus existing model state. Default owned mode
still selects the existing three actual512MiB private allocations.

A further source audit found mutually contradictory constructor checks: GR's
native524288-byte increment appeared in the actual family requests and TOTAL,
but was absent from NEW. `12b310f` includes it in NEW and defines TOTAL as OLD+NEW,
so both checks describe the same owner vector. Exact admission is preserved;
this does not waive mismatched requests or claim whole-model construction.

### Validation and remaining gate

Validation receipts are under `/media/jmicko/faststore/kuno-development/`:

- `strata-shadow-connected-tests-20261014.log`: three registry/layout tests pass,
  including retired identity/counters, failed extension atomicity/metadata peak,
  all and only lent roles, exact nonoverlapping ranges and mixed-shard identity.
- `strata-shadow-prompt-tests-20261014.log`: all15 focused prompt tests pass.
- `strata-shadow-plan-tests-20261014.log`: allfive original source/window and
  empty/partial/full-cache reservation tests pass.
- `strata-shadow-close-tests-20261014.log`: erasure/restoration order, full receipt
  and fault/panic rejection pass (caught synthetic panic text is expected).
- `strata-shadow-pricing-tests-20261014.log` and
  `strata-shadow-placement-tests-20261014.log`: both pass, checking absent loan
  owners/bytes versus default owned allocation and exact inverse placement
  through repeated adaptive swaps.
- `strata-shadow-gr-owner-tests-20261014.log`: actual MAX8 GR owner-envelope
  test passes after the aggregate correction.
- `strata-model-rebase-accept-tests-20261014.log`: allfour registry/layout tests
  pass, including active/selected/unselected identities, atomic duplicate reject,
  token metadata accounting, extension/detach exclusion and abandoned rebase.
- `strata-model-rebase-accept-build-20261014.log`: final core release build passes.
- `strata-model-rebase-accept-final-core-clippy-20261014.log` and
  `strata-model-rebase-accept-off-clippy-20261014.log`: final strict core and
  ordinary feature-off Clippy both pass. Two transient dependency documentation
  warnings in the first rebase check were fixed; final checks are clean apart
  from Cargo's existing duplicate-binary-path warning.

These use `--locked --release --manifest-path
apps/kuno/benchmarks/native-text/Cargo.toml --bin kuno-strata-rust`;
core feature `strata-intel-core`, ordinary feature `strata-rust-port`, and Clippy
`-- -D warnings`. No unprotected inference invocation is part of these checks.

No shader arithmetic changed, so the preceding308-variant offline shader receipt
was not repeated. No live allocation/model/loan refresh/erasure/restoration,
whole-model numerical/state/recovery or speed execution was performed. The historical
228 full-core failures still require a complete audit; these focused checks do not waive
them. No installed services/model residency, API/Vashti, install/release, driver
or reset change. Canonical checkout and inherited bytecode remain untouched.

Next finish actual runtime context/configuration and pre-device HOST grants,
cache-budget placement, conditional STAGE8/stream-all/big-ring/auto-chunk policy,
then complete arbitrary input/sampling, full captured branch/state-repair and
source/unit-failure audits. The new loan source/constructor/controller path is
compiled but current graph/request permits remain4096, so no supported long-
context invocation or physical parity is claimed. Run the protected complete
engine numerical/completion/state/recovery gates before opening speed testing.
Required Vashti changes: none.


## Native prompt prefix and final decode boundary (2026-10-14)

Source `97e22f9`, after `8b503e2`/handoff `0b25010` on isolated
`codex/strata-rust-port`. This corrects the actual selected controller/math path;
it does not remove the remaining context/sampling/configuration gates or claim
whole-model numerical acceptance. Both GPUs remain reserved for the user.

Pinned upstream `sycl/src/program/generate.cpp:11520-11624` sets
`n_batched=n_prompt-1`, uses `request_chunk` (round up256, capped by the configured
chunk), teaches MTP's every prefix residual with the actual next prompt token,
and restores lent cache slots before returning to ordinary decode. Its
`11837` decode sampler is indexed by that ordinary input position. Equalized
chunks are an opt-in branch at5804-5828, not the default.

The old connected Rust controller batched ALL prompt inputs, gave its last
chunk a prompt-precision vocabulary head, and returned that chunk's hidden row.
That executes different arithmetic for the final prompt input even with the
same tokens/settings. This source removes that alternate head rather than
retaining it as a partial-performance option:

- `PromptPlan` selects exactly the firstN-1 inputs and the native default rounded
  chunk, bounded by the actual2048-row stage. Each chunk supplies equally sized
  input/next-token teacher slices, with the final teacher containing the last
  original prompt token. A one-token prompt has no batched prefix or loan
  execution. Construction still reserves its default owned prompt roots even
  for that case; request-sized construction/pricing is pending context work.
- The request-scoped PLE prefetch reads only this batched prefix. One actual
  borrowed/owned Scope spans its chunks and MTP teaching. The worker joins,
  private scratch closes/inspects and borrowed weights restore BEFORE decode.
- `prefill_prompt_chunk` is now prefix-only and returns no vocabulary output.
  It preflights exact teacher length/draft position before executing target math.
  The old prompt-output_hc/head, full host logits readback and alternate carry
  extraction are removed from this selected path. GDN/QSA/PLE/fault/state commit
  remain coupled to the same layer-major prefix execution.
- The final original input always goes through the existing normal count1
  `prefill_chunk_with_carry` graph, exactly once. Its actual output position/count,
  compact pick/carry, full fault/state acceptance and final committed cursor are
  checked before normal request output/reuse. MTP is already atN-1; no extra or
  missing teacher cell and no provisional target row is emitted as a result.

Validation logs under `/media/jmicko/faststore/kuno-development/`:

- `strata-native-prompt-boundary-plan-tests-20261014.log`: schedule/teacher checks
  pass for1,2,30,35,255/256/257/258,2048/2049/2050,3073,4095/4096 inputs. Every
  prefix input is consumed once, every R_i pairs with the exact next token,
  the final original input is excluded from batching, and malformed ranges/
  lengths/overflow reject. This uses the actual controller's plan and slicer.
- `strata-native-prompt-boundary-final-io-tests-20261014.log`: the actual I/O
  authority accepts prefix chunks then count1 final input in Prefill, followed
  by output scope/early-EOS closure. Its obsolete257-row rejection now rejects
  CAPACITY+1; valid2048-row prompt chunks are no longer treated as invalid.
- `strata-native-prompt-boundary-accept-tests-20261014.log`: existing accepted-
  prefix/rejection/EOS test passes for every supported verification width.
- `strata-native-prompt-boundary-build-20261014.log`: core release build passes.
- `strata-native-prompt-boundary-final-clippy-20261014.log` and
  `strata-native-prompt-boundary-off-clippy-20261014.log`: strict core and
  ordinary feature-off Clippy pass. Formatting/whitespace pass. Existing Cargo
  duplicate-binary-path warning remains; no new compiler warning.

Commands use `--locked --release --manifest-path
apps/kuno/benchmarks/native-text/Cargo.toml --bin kuno-strata-rust`, core feature
`strata-intel-core` or ordinary `strata-rust-port`, and Clippy `-- -D warnings`.
No shader arithmetic/ABI changed; the prior offline shader receipt was not
repeated. These host checks are not a whole-model or full-suite run.

The host schedule/permit checks do not execute math or prove kernel precision,
physical erasure/restoration, MTP numerical alignment or whole-engine output.
No GPU/device/model execution, services/admission/residency changes, API/Vashti,
installation/release or speed trial. Prior full-suite failure audit is still
unfinished; no old failure is waived. The complete source/physical acceptance
gates remain closed, and runtime context/sampling/conditional-ring/branch/repair
work is still required before any throughput trial. Required Vashti changes:none.

For the later sampled path, follow the pinned Intel
`sycl/src/kernels/cuda/sampler.dp.cpp` and `sycl/include/strata/kernels/sampler.hpp`:
original penalty order, split4096/top64 GPU candidates and ordered F32 tail,
Philox4x32-10 keyed by seed/counter, per-verification-row penalty history, and
coupled draft staging/sample/probability/target acceptance. A CPU sampler or
random draft alone does not finish this source-to-source dependency. This pass
inspected those actual files but did not claim to implement sampling.

## Native sampler source and request-data boundary (2026-10-14)

**Source `267a71b`: sampler component only; NOT connected target/draft sampling.**
This extends `97e22f9`; it does not satisfy the sampling or arbitrary-request completion row.
The existing controller/target packet/MTP head still take their earlier greedy
paths. No GPU/model execution or speed comparison was conducted. The user owns
both GPUs until availability is reconfirmed.

Pinned sources inspected: `sycl/include/strata/kernels/sampler.hpp`,
`sycl/src/kernels/cuda/sampler.dp.cpp`, `sycl/src/kernels/sampler_parity.cpp`,
`sycl/include/strata/core/coupled_draft.hpp`, and the actual call sites in
`sycl/src/core/{verify,mtp}.cpp` / `sycl/src/program/generate.cpp` at
`fb58e0dbc8399662c0e47c76578c6e878b14f6cf`. MIT notice is retained/extended in
`strata-port-NOTICE.txt`.

### Implemented component, not a connected engine claim

`strata_sampler.rs` and its four WGSL sources provide:

- Full original248320 vocabulary; one1024-thread workgroup per greedy row,
  including two subgroup reduction levels. Penalty membership uses the native
  vocabulary bitmap and exact counts only on hits. A separate no-penalty
  specialization avoids reserving the full31KiB bitmap. WGSL requires a
  nonempty static array, so that specialization retains one unused4-byte word
  (native SYCL's dynamic local bitmap has zero bytes in this case).
- Sampled split top-k:61 blocks of4096, four32-lane subgroups per block,32
  register-held values per lane and separate even/odd selection chains. A
  subgroup merges four lists using private cursors. One32-lane workgroup per
  row merges the61 lists, with two private list cursors per lane. There is no
  workgroup barrier per selected token and no partial-workgroup barrier.
- Penalties precede top-k, then top-p, min-p, temperature and inverse-CDF draw.
  Tail exponentials and quotients are distributed among32 lanes. Lane0 alone
  performs ascending F32 sums/cumulative scans. Both selected id and its final
  probability are returned in the private output, including probability1 for
  greedy. Finite logits/normalizers join the sticky fault; this is not a
  substitute for the existing target/state completion proof.
- Pinned Philox4x32-10 uses upstream's actual `0x9E3779B9` / `0xBB67AE85`
  multipliers, not the more familiar Random123 constants. Integer high products
  use a checked16-bit partial-product decomposition; draw counters use full
  low/high carry and24-bit uniforms below1.
- `penalty_rows` packs the last history tokens from consumed tail plus each
  target row's own input prefix, with front `-1` padding. `coupled_base` is its
  row0. `coupled_counter(cell)` is cell+1, the target row checking that draft,
  rather than the cell's predicted-token position cell+2. Helpers/staging wipe
  host storage on drop; no diagnostic/persistent token history is added.

The GPU launch contract uses six disjoint, same-stream private views in order:
logits, per-row history, split scratch, strided `[id, probability-bits]` output,
sticky fault,128-byte request parameter storage. The last buffer is **device
storage**, not shader text, kernel identity or an INFO/uniform literal. Request
seed/counter/settings must be refreshed there at execution; native graph replay
must never reuse a captured literal draw. Kernel identity contains only the
static algorithm/penalty specialization. Launcher performs geometry/extent/
non-overlap checks but is unsafe: its eventual owning caller must authenticate
complete arena grants and exact parameter-buffer contents, stage them through
owned private I/O, retain owners through both queue joins, and erase them.
It allocates no GPU buffers, samples no host logits and writes no persistent
request data.

Sampled scratch is rows*61*64*8 bytes (31232 for one row,249856 for eight);
greedy needs only a dummy4-byte candidate view. Per-row history is at most
4096 int32 slots. Values are fixed to this original model/currentMAX8 scope,
not an assertion of general-vocabulary or arbitrary-context support. Runtime
request grants, actual buffer carving and resource admission remain to connect.

Substitution requiring final review/numerical acceptance: WGSL subgroup max plus
minimum tied token id implements the native finite-logit total-order reduction
instead of spelling SYCL's shuffle tree. Grouping, private cursor ownership and
ordered sums are preserved; device transcendental/rounding behavior and selected
probabilities still need physical comparison. This component is not evidence of
bitwise device equivalence, correct whole-model output or graph recovery.

### Next connected block and remaining audit

Do **not** finish by selecting a CPU sampler or merely binding these shaders to
the old greedy packet. Connect these dependencies together:

1. Real pre-device MAIN/HOST/private-I/O prices, owning request buffers and
   bounded wiping history/configuration. The component's unsafe launcher is
   deliberately not called with fabricated owners or ordinary uploads.
2. Actual target logits -> per-window parameter/history staging -> sampler ->
   authenticated compact output/carry/fault packet. Pinned verify.cpp:2166 draws
   target row t with counter pos0+t, after the main verification graph; source
   explicitly keeps this call outside the recorded head to avoid stale draws.
   Greedy-with-penalties must use the same history path.
3. Actual MTP head and coupled round/step inputs: pinned mtp.cpp::setup_coupled,
   set_draft_sampling/history, capture_round, capture_step and record_rest.
   Refresh device parameters and the base ring before the round; read the
   actual cell position from its device step record; draft j reads
   ring[cap+j-h,cap+j), and writes its actual id into ring[cap+j]. Probability
   is part of its output. Keep original full Q2 head/source authority. The
   request helpers here do not implement this GPU ring/stage/publish chain.
4. Existing exact-match target acceptance remains authoritative under sampling
   (coupled_draft.hpp): a draft may only increase accepted length when it equals
   the target's sampled token. Consume only the accepted input prefix in history
   and state; repair rejected GDN/QSA/PLE/MTP state and keep absolute draw indices
   invariant across window sizes. Do not count proposed suffixes as consumed.
5. Extend actual graph ownership/branch recordings, INFO/host grants and final
   erasure paths, plus CPU/source failure audit. Then protected numerical/model/
   state/recovery acceptance, still before speed trials. No frontend/API change.

The prior full-core suite's228 failures remain unaudited; this source receipt
neither skips them nor treats focused helper/shader checks as a full suite pass.
Validation receipts are recorded with the source commit in the coordination
handoff after the CPU checks complete.


### Sampler source validation receipt

Final CPU-only receipts in `/media/jmicko/faststore/kuno-development`:

- `strata-native-sampler-final-tests-20261014.log`: all7 focused checks pass.
  Coverage is independent full-vocabulary sorting versus partition/merge,
  penalties/ties/partial last block, ordered scalar tail/probability invariants,
  pinned Philox draws/high-product/counter carry, exact device payload and
  storage reflection, input rejection/extent overflow, and all supported target
  row histories/coupled base/cell-counter relationships. Four sampler variants
  parse/validate and lower to SPIR-V. These are CPU source/geometry checks,
  **not execution of GPU sampling against an independent device oracle**.
- `strata-native-sampler-final-build-20261014.log`: core release build passes.
- `strata-native-sampler-final-shaders-20261014.log`:312 WGSL variants validate
  and lower, including the4 new sampler variants. The122 existing physical
  pointer checks and32 cooperative-matrix/layout variants pass external SPIR-V
  validation as well. No Vulkan device is created by this command; main returns
  from `--validate-port-shaders` before selecting/initializing a device.
- `strata-native-sampler-final-core-clippy-20261014.log` and
  `strata-native-sampler-final-off-clippy-20261014.log`: strict core and ordinary
  feature-off Clippy pass. Formatting and staged whitespace checks pass. The
  pre-existing Cargo duplicate-binary-path notice is unchanged.

Same locked release manifest/bin as the preceding receipt, test filter
`native_sampler`, features `strata-intel-core` / `strata-rust-port`, Clippy
`-- -D warnings`. Full shader validation uses the existing pinned2026.1 tool:
`PATH=/media/jmicko/faststore/kuno-development/spirv-tools-matrix-20261014/build/tools:$PATH`.
Its revision/version is in `strata-native-sampler-validator-20261014.log`.
The first full-set attempt with the distro2025 validator stopped at an existing
BF16 capability it does not recognize; the final receipt uses the already
built pinned official tool documented above. No capability was skipped or
removed and no installed tool/service was changed.

Canonical checkout was verified clean and left untouched; inherited bytecode
in the isolated checkout is preserved. No public API/consumer, service,
installation, residency/admission, release or live-speed change. The goal stays
active; completeness and protected numerical/state/recovery gates remain open.

## Connected request sampling owners and target/MTP hooks (2026-10-14)

Source `ab0b15c` follows sampler component `267a71b`. Source connection is now implemented;
GPU numerical/queue/erasure acceptance, complete Strata confidence-policy parity,
arbitrary-request invocation and the full rewrite remain **unproven/incomplete**.
The user still owns both GPUs. No GPU/model run, install, service change or speed
trial belongs to this receipt.

### Actual connected calls and ownership

The selected global controller's internal `Request` now takes wiping sampler
parameters. It creates `ResidentModel::begin_sampling_request` after prompt-only
loans/teaching finish, before the final original input is normally decoded. The
current fixed public harness explicitly requests greedy behavior; the internal
request type supports the sampler chain without adding an HTTP/API/UI contract.
There is still no general arbitrary-input command/worker acceptance claim.

`strata_global_moe::Workspace` allocates seven additional, exact complete arena
owners and includes them in `requests_for_context`, `main_bytes_for_context` and
`MAIN_BYTES`. Sizes are32768/131072/262144/32768/32768/32768/32768 bytes,
**557056 MAIN bytes** total: parameters, target-history rows, split scratch,
id/probability pairs, coupled ring/probabilities, step controls and round base.
The constructor is unsafe: its sole-worker caller must retain both original
models, complete arena ownership and pre-device grants, without reset/close or
client replacement, until the lexical scope joins/closes or retires. The actual
controller keeps those exclusive model/executor/draft borrows throughout.
The roots retain their original arena IDs/epoch and are never public model weights,
cache loans, borrowed source pages or uncharged implicit allocations.

The new `strata_sampling_request` Scope validates all seven original layouts,
full spans, distinct allocations, common stream, target graph and optional MTP
binding, protected owning-I/O audit mode, and bounded prefix/settings before
publishing TLS state. It is non-Clone, `must_use`, and explicitly **not Send/Sync**;
a request guard cannot move away from the TLS history it owns. Consumed tokens
and staged window have bounded wiping capacities and request positions wipe on
drop. Parameters retain their non-Debug/non-serializing wiping contract.

Private uploads use the existing **full-owner** `write_private` interface;
subviews do not acquire write/free authority. Main roots do not grow during a
request. Parent pre-device host accounting adds2MiB for request/history/staging/
inspection. Target runtime accounting and the MTP runtime envelope explicitly
include3932160 bytes: source/destination/independent inspection allowances for
one maximal draft/target window and full-owner close inspection. No existing
expert/INFO credit is reused; sampler/bridge dispatches have no request INFO
literals. The current allocator's32KiB minimum means parameter/control uploads
include authenticated zero padding, a backend granularity substitution.

### Target and history commit

`Scope::stage_target` records the real proposed *inputs*, absolute start and
count. `resident_target_rows::Owners::read` authenticates original logits,
carry, fault and packet as before. Inside the original protected packet-I/O
closure it calls `request::target_pick`: build the pinned per-row histories,
refresh full private history/parameter owners, then dispatch the real GPU
sampler. Its output is consumed by the new separate selected-token/carry packet
kernel, with unchanged compact header/carry geometry and fault fence. It checks
ids/probabilities and zeros inactive rows. The old greedy packet remains for
other laboratory calls outside this owning request; the selected controller
uses the new path even for the first normal input/greedy generation.

No host logits or host fallback select the token. Existing `NativeTargetRows`
selected-token accessor feeds the controller's exact-match verification. Only
AFTER target acceptance/state commit does `Scope::commit_target(n)` append the
accepted input prefix to its consumed history and wipe the provisional window.
Rejected suffix inputs and the newest predicted-but-not-fed token are excluded.
The next target window draws at its real absolute start, regardless of drafts.

### Coupled MTP connection and explicit substitution

Before proposing each window, `Scope::begin_round` uploads the native padded
4096-slot consumed-tail-plus-confirmed-token base and request chain to its real
round owner. The last slot always holds the actual next input, including when
penalties are disabled. Normal greedy requests preserve Strata's uncoupled
argmax drafting; sampled requests use the following hook in
`ResidentMtp::finish_gpu`, after the original full Q2 head projection:

1. Prove the same client/original MTP binding; reject subset-head use in this
   owning full-vocabulary request. Stage the current cell and chain index in a
   complete private control owner. The GPU Prepare bridge reads the **actual
   native-front device position table**, checks the expected current cell and
   round-relative index, and sets the sampler counter to device-cell+1.
2. The first step copies the request/base to device parameters/ring; subsequent
   steps keep the ring and read `[cap+j-h, cap+j)`. Prepare verifies the actual
   GPU-fed `B::Token` against the predecessor in the ring. Each parameter word
   has one writer, with explicit storage/workgroup ordering before history reads.
3. Dispatch the same original-vocabulary sampler against actual MTP logits.
   The Publish bridge checks the sampled id/probability/sticky fault, writes
   `B::Greedy[0]` for the existing draft packet/follow-step consumers and the
   actual token to ring[cap+j]. Probability bits are retained at ring[4104+j].
   The subsequent queued step consumes this actual GPU token/residual.

`source fb58e0` carries the current cell in `step_rec[0]`; Rust's grouped front
already writes its authenticated position table, which is the selected device
position input here. The separate private control supplies the expected index,
not the draw value or seed; the GPU reads/verifies its table value before setting
cell+1. This preserves the counter/data dependency with a different record layout.
Full native round/step branch and queue acceptance is still required: compilation
of this substitution is not physical proof of correct front/head execution or
replay. Request values stay in device storage, not generated source/kernel keys.

The existing accepted-prefix target/GDN/QSA/PLE and MTP repair remains in force;
this hook never approves a draft on its own. Confidence probabilities are produced
and checked, but the original host confidence/early-stop/`chain_launch` consumers
are **not yet ported**: `DraftWindow` still returns ids, and the controller still
uses the earlier cost/suffix policy. Greedy MTP's original top-probability chain
also needs its source/consumer audit. Do not label this complete sampler/MTP
policy parity until those dependencies and numerical/state behavior are proved.

### Normal close, failures and remaining gates

After the final accepted output/cursor check and before the original language
I/O close, `Scope::close` clears all seven complete main owners, reads each via
protected cleanup, checks every byte/padding is zero and wipes host inspection
bytes. The parent's existing close then erases/audits transient I/O. Abandoned or
failed scopes retire the model context; the original arena retains root ownership
and its established failure cleanup rather than freeing borrowed views or
publishing clean recovery without proof. Device close/recovery has not been run.

CPU state tests use the actual staging/commit functions: all supported widths/
accepted lengths, rejected/duplicate commit, untouched preflight failure, exact
owner rounding and global request/MAIN sums. Existing scalar sampler/history/
Philox/ABI checks remain. Added bridge/selected-packet tests parse/validate/lower
without a device. These prove source/control relationships only. Whole-model
outputs, captured replay changes, actual zero inspection and GPU memory recovery
still require protected execution after GPU availability is reconfirmed.

Continue confidence/captured branch/repair and failure audits, arbitrary runtime
input/context and pre-device grants/cache policy, conditional ring/stream-all and
remaining source completeness work. Prior full-suite failures remain unresolved;
no unit failure is waived. No speed trial until the full connected-source and
whole-engine correctness gates pass. No Vashti changes are required by this
internal engine block.


### Connected sampling CPU/source receipts

Final receipts under `/media/jmicko/faststore/kuno-development`:

- `strata-sampling-connected-final-tests-20261014.log`: all10 focused checks
  pass (`native_sampl` filter). Seven sampler/ABI/history/Philox checks plus
  actual request stage/commit transitions for all widths/accepted lengths,
  preflight/price/layout invariants and bridge/selected-packet lowering. These
  do not construct a GPU Scope, execute sampled math or prove recovery.
- `strata-sampling-connected-final-build-20261014.log`: core release build passes.
- `strata-sampling-connected-final-core-clippy-20261014.log` and
  `strata-sampling-connected-final-off-clippy-20261014.log`: strict core and
  ordinary feature-off Clippy pass after the final owner-lifetime/thread guards.
  Formatting and whitespace pass. Cargo's existing duplicate binary path
  notice remains; no new source warning is accepted.
- `strata-sampling-connected-final-shaders-20261014.log`:314 regular shader
  bodies parse/validate/lower, selected-packet controller validation passes,
  and122 physical-pointer /32 cooperative-matrix checks pass. Uses the existing
  pinned2026.1 validator PATH described above. The focused selected-packet
  test also lowers it to SPIR-V. Independent offline checks validate the new
  Prepare/Publish/selected modules with `spirv-val --target-env vulkan1.3`.

Locked release manifest/bin/core/off feature commands are unchanged from the
preceding receipt. The new binary's source-only shader option exits before any
device is selected/initialized. No GPU/model process was started. Source and CPU
checks are authoritative progress; source completion, whole-model correctness,
physical root/I/O erasure and GPU recovery are not proven. Prior228 full-suite
failures remain an explicit audit item. Goal stays active. Canonical checkout
was checked clean, and inherited isolated bytecode is preserved.

## Native MTP confidence producers and serial consumers (2026-10-14)

Source `8f1377c` extends `ab0b15c`. The source now connects native-style greedy top
probability and coupled sampled probability to real draft packets, threshold
stopping and target-window selection. It is not whole-rewrite completion or GPU
acceptance. Both GPUs remain with the user; this pass is CPU/source work only.

Pinned references at `fb58e0dbc8399662c0e47c76578c6e878b14f6cf`:
`sycl/src/kernels/cuda/verify_kernels.dp.cpp::row_top_prob_split_kernel`,
`row_top_prob_kernel`, `multi_block_head_ops` / `argmax_rows_wanted`;
`sampler.dp.cpp::sampler_greedy_kernel` / coupled sampled tail;
`sycl/src/core/mtp.cpp::record_rest` / `draft` / `chain_launch` and
`sycl/src/program/generate.cpp` serial `spec_min_p` loops. Intel builds default
to the multi-block top-probability implementation. CUDA-architecture-specific
argmax dispatch is distinct; this port selects the native1024-thread no-penalty
fallback and retains a selection/numerical audit, rather than claiming a CUDA
architecture branch is automatically appropriate to Intel.

### Real head producers and existing private owner

`strata_mtp_probability` now supplies two real consumers of the original full
Q2 MTP head in `ResidentMtp::finish_gpu`:

- A1024-thread/no-penalty argmax, native32-lane strided scan, shuffle-down32
  reductions and two shared/subgroup reduction levels. Lower token id wins
  finite ties. Original private logits/token/fault roots stay authoritative.
- The native8x128 top-probability shape. Virtual thread b*128+j visits vocabulary
  elements at stride1024, sums F32 exponentials relative to the selected maximum,
  then XOR-reduces offsets16/8/4/2/1 within each32-lane subgroup. Each of32 warp
  sums has its own slot. The last CTA after an8-way atomic counter sums those
  slots in ascending order and writes reciprocal total, then resets the counter.

Scratch reuses the original complete private `B::Candidates`32KiB allocation:
first256 bytes for counter/padding/32 parts, a disjoint4-byte view at byte256
for probability (`WORD=64`). No new MAIN allocation or constructor credit is
introduced. Cell start already clears Candidates in both fused/scalar and
normal/follow paths; neither native front/rest nor the new argmax writes it.
The top-probability kernel initializes its output to NaN before the counter
publication, so an absent final producer cannot silently become probability0.
Native greedy head selection plus probability remains two dispatches.

Coupled sampled MTP keeps the previously connected filtered sampler and device
ring. Its Publish bridge additionally copies the actual selected probability
bits into Candidates[64]. It does not run unfiltered top probability on sampled
logits, overwrite the chosen token with an argmax, or replace probability by1.
Thus both paths feed one checked packet consumer, with their distinct meanings:
full-vocabulary argmax mass versus the sampled chain's final filtered mass.

Necessary Vulkan/Rust substitutions remain explicit:

- WGSL `exp` represents SYCL `native::exp`; no bitwise transcendental equivalence
  is established. Physical numerical comparison/rounding review is outstanding.
- Native volatile float warp parts are atomic u32 bit stores/loads here. Release
  precedes the counter RMW; acquire follows it before the final32-part read.
  WGSL storage barriers lower to Workgroup execution + **Device** memory scope,
  AcquireRelease|UniformMemory. The CPU SPIR-V test verifies both device barriers
  and the atomic increment exist; that is not physical cross-queue/device proof.
- The probability view aliases a disjoint part of an existing complete private
  owner. It gets binding/read authority only; original arena cleanup retains
  the entire allocation and its admission. No subview may acquire free authority.

### Packets, native threshold and suffix interactions

`DraftOutput` now owns an optional wiping scalar confidence, with an explicit
error for unsupported legacy profiles. The existing full64KiB RawQuery packet
uses one additional logical word for native probability; readback allocation,
carry/token/fault offsets, owner lease and padding checks remain. CPU parsing
checks the selected native format, finite[0,1] probability, token range, sticky
fault and all unused bytes. The legacy parser/format remains separate.

`DraftWindow` similarly returns bounded wiping probabilities with its actual
proposed ids. Its existing32KiB packet uses two ROWS-wide logical planes: token/
fault and probability/reserved, with inactive rows/padding zero. Both scalar
packet generation and queued window generation have checked/unchecked, mode-
specific CubeCL-to-WGSL validation; the raw head shaders validate/lower separately.

`draft_window_with_min_probability` mirrors pinned serial `draft()`:
minimum0 queues the complete requested chain before one final read/fault fence.
A positive threshold generates draft0 and launches j only when j-1's completed
probability is >=minimum. The first low draft is returned for observation; the
controller's `confidence_window` excludes it from the target inputs. Threshold
comparison includes equality. A legitimate rounded probability0 is accepted;
NaN/range/fault/sentinel checks detect an absent or invalid producer instead.

The selected controller uses the request's native maximum window, capped by
remaining output/context, rather than the earlier custom adaptive MTP-length
chooser. It produces the confidence-bearing chain BEFORE suffix-policy choice,
then chooses the target prefix from probabilities. A lookup candidate whose
first token agrees with the original draft can still win the existing suffix
policy even when MTP confidence is low. Only chosen actual inputs reach target
verification, whose exact-match acceptance remains unchanged.

Native accepted-cache repair now permits an accepted suffix prefix longer than
the short confidence-stopped MTP chain. The real Strata-Q2/core entry selects an
explicit native plan mode: retain the correct first cell, rewind provisional
later KV and append actual accepted tokens with their actual target residuals.
All count/context/input/finite/source checks remain. Legacy Q8/frozen repair
constructors keep their older draft-count boundary. The plan never approves a
token or substitutes a draft residual for an accepted target residual.

### Zero-probability handling and remaining source gates

F32 sampled distributions may return probability0 (e.g. rounded tail fallback),
and native confidence0 is a valid stop signal. Packet/publish checks now accept
finite[0,1]. To preserve missing-writer detection, target Scope resets each
active output pair to no-token/NaN through its complete private output owner,
and the GPU draft Prepare initializes the single output pair likewise. Invalid
sampler geometry/subgroup support sets the sticky fault rather than silently
returning. The output initialization adds one full32KiB private upload/target
window; its source/destination/inspection cost is explicitly added to the
pre-device request/MTP runtime envelope:4063232 bytes total. MAIN557056 and
host2MiB grants remain unchanged; no prompt, seed, logits or probabilities are
written to disk/logs/cache keys.

Still outstanding: force-token/nonblocking pipelined `chain_launch` event and
partial-output publication, stable complete native round/step graph branches,
and the exact placement/grouping of accepted-front catch-up and next draft rest.
This serial connection still repairs its accepted front then starts the next
cell; source-schedule parity with native capture_round needs its own connected
audit/port. Existing cost/suffix-policy fidelity and no-MTP lookup branches also
remain audit items. Do not close those rows because probability math compiles.

Also retain arbitrary request/context/grants/cache sizing, conditional ring/
stream-all, source/228 prior unit-failure audits and protected whole-model
numerical/state/erasure/recovery gates. No live speed trial until full source
completeness and whole-engine correctness pass. No public API/Vashti, service,
installation, residency/admission or release changes are part of this block.


### Native confidence source validation receipt

CPU-only final receipts under `/media/jmicko/faststore/kuno-development`:

- `strata-mtp-confidence-final-tests-20261014.log`:8 focused confidence tests
  pass. Covers native optional scalar/window packet formats and full padding,
  zero/NaN/range behavior, threshold equality/first-low exclusion, every supported
  width, suffix repair outgrowing a short native draft, head shader lowering and
  the two actual Device/AcquireRelease/UniformMemory barriers plus atomic RMW.
  The scalar1024-thread/XOR32/ordered32-part oracle agrees with an independent
  full-vocabulary reference within its stated CPU tolerance. Actual packet
  CubeCL expansion validates checked/unchecked and legacy/native modes, including
  the dynamic window index. This does not run kernels or prove GPU math/order.
- `strata-mtp-confidence-final-sampler-tests-20261014.log`:10 sampler/request
  regressions pass after zero-probability/sentinel/runtime-quote changes.
- `strata-mtp-confidence-final-build-20261014.log`:core release build passes.
- `strata-mtp-confidence-final-shaders-20261014.log`:316 ordinary shader bodies
  validate/lower, with122 existing physical-pointer and32 cooperative-matrix
  validations. Uses the documented pinned2026.1 validator. Separate offline
  `spirv-val --target-env vulkan1.3` checks accept Prepare/Publish/selected packet
  and both new head kernels. No GPU/device is initialized by these checks.
- `strata-mtp-confidence-final-core-clippy-20261014.log` and
  `strata-mtp-confidence-final-off-clippy-20261014.log`: strict core and ordinary
  feature-off Clippy pass. Formatting/whitespace pass. Existing Cargo duplicate
  binary path notice remains. The targeted CubeCL `collapsible_if` allowance
  keeps compile-time mode specialization separate from runtime device control;
  both generated modes are validated. No broad warning suppression was added.

Commands retain the same locked/release manifest/bin/core/off features; test
filters `native_confidence` and `native_sampl`. Earlier fixture compile/entrypoint
and Clippy failures were corrected before these final receipts; no source or
speed-based rollback. No full-suite run/waiver of the prior228 failures, live
model/numerical/recovery test or throughput trial. Source code and compiler
checks are progress, not completion of the faithful whole-engine goal.

Next connect native round/step/catch-up state/branch scheduling and applicable
pipelined chain event consumers, then runtime input/context/cache policy and all
remaining completion rows. Continue CPU/source work while the user owns BOTH
GPUs; obtain availability before any device/model or residency/admission work.
Canonical checkout remains untouched/clean; isolated inherited bytecode is
preserved. No service, public API, Vashti, install or release change. Goal active.


## Grouped accepted MTP front and next draft rest (2026-10-14)

Source `f671543`, isolated `codex/strata-rust-port`, after `8f1377c`. This is a
connected scheduling block, **not complete captured-graph/device handoff parity**.
It is selected in the actual global controller, not a default-off helper.

### Source mapping and dependent consumers

Pinned Intel `sycl/src/core/mtp.cpp::capture_round` (1227), `capture_step` (1280)
and `draft` (1534) group `T=a+1` accepted rows, copy row a's R/injection/mixed to
row0, requantize its mixed input, then execute one rest/head. Later drafts use
one front/rest each. `sycl/src/program/generate.cpp` (12335) supplies the actual
previous target predictions, p and a after target commit. Rust now follows that
row grouping in `strata_global_controller.rs`, `resident_mtp_window.rs` and
`ResidentMtp::native_selected_rest`:

- A continuing speculative round moves its existing actual target prediction/R
  vectors into one bounded request-local `AcceptedRound`. No second prefix mirror
  is allocated. Start/old cursor/model binding and vectors wipe on drop. No
  predictions, residuals, probabilities or coordinates are persisted or logged.
- At the next entry, the controller checks original binding/cursor, accepted row
  count and confirmed last token, then begins the existing coupled sampler round.
  The native MTP entry validates every real token/carry/cell before consuming the
  completed-forward stamp. It supports all accepted counts1..8, including a
  suffix window longer than a confidence-stopped draft chain.
- If target window p accepts n inputs, the retained known first draft cell is
  p-1. Rewind removes provisional cells from p onward; one front call appends
  **predicted[r], R[r] at cell p+r**, r=0..n-1. The final cell p+n-1 includes the
  newly confirmed next token. These are not the old repair's accepted inputs
  paired with the preceding residual. Rejected target rows never enter catch-up.
- `strata_mtp_front::Kernel::Select` copies the selected final row's full10240 R,
  four injection and2560 mixed values together into row0 of the existing complete
  front owners. Source/destination intervals are disjoint; each plane has one
  read_write binding and each destination one writer. Geometry, full lengths and
  finite checks feed the sticky fault. Count1 needs no selection dispatch.
- The selected mixed input is requantized into the existing Q8_1 PLANES before
  the rest's query projection reuses it. The actual selected token remains in
  B::Token, the absolute device POS table selects the new cell, and sampled draft
  draws still use that cell+1 and the actual request penalty ring. Subsequent
  steps consume the preceding GPU token/residual, never a host substitute.
- Zero probability threshold queues the first grouped front/rest and subsequent
  steps before the original final full-owner packet/fault fence. Positive
  threshold joins the first grouped result, then gates each following step on
  its actual probability. Target prefix verification/commit is unchanged.
  A remaining count1-only/final round uses the existing exact cache repair;
  there is no next proposal needing a grouped front. EOS/output-limit exits do
  not leave an unconsumed pending prefix.
- `start_at` distinguishes the selected input cell from the old completed cache
  cursor, so a valid rewind from a full cache is not mistaken for execution at
  cell4096. All original full owners, admission, guards, retirement/failure and
  closing authority remain in their original arenas. No new MAIN, runtime or
  host reservation is introduced. Existing per-cell512 INFO enforcement and
  queued count*INFO precharge remain; physical dispatch-budget acceptance is open.

The first prompt-produced proposal still uses the same count1 native front/rest.
The old eager accepted n-1 front plus separate next-cell front is no longer used
between continuing speculative windows in the selected controller.

### Explicit remaining substitutions and gaps

The grouped dataflow is now connected, but **the target-to-MTP residual source
still differs from Strata**. Upstream `Verifier::final_R_all()` is its device R_
owner (`verify.hpp:251`, `verify.cpp:507/544`); serial `mtp.bind` uses that owner
and the round graph copies its rows directly. Rust still decodes the original
full private verifier packet to bounded wiping host carries and uploads them to
its front CARRY owner. That round trip is not required by the model. Replace it
with a lifetime-bound original device-source grant and authenticated GPU copy;
do not invent ownership/erase authority from a slice or keep an obsolete source
across target scratch reuse. The scalar host carry path remains for existing
laboratory interfaces; it does not establish native placement parity.

The selected row copy and requantization are Rust/WGSL substitutes for native
`copy_row_to_first`/`native_quantize_q8_1`. Their offline lowering is checked;
physical arithmetic, state, barriers, failure erasure and recovery are unproven.
Native stable round/step branch captures/warm-up, proper mapped staging/event
publication, applicable nonblocking/force-token chain consumers and overlap with
commit remain open. Existing bounded Vulkan node replay is not proof of those
whole native branch captures. No incidental timing changed the port direction.

### CPU-only receipt

Final logs in `/media/jmicko/faststore/kuno-development`:

- `strata-mtp-grouped-round-final-tests-20261014.log`:4 checks pass. Covers all
  accepted widths/short-chain bounds, invalid token/carry/context/completion
  preflight, original binding/cursor/confirmed entry and all front shader lowering
  plus disjoint selected-row spans. These are source/CPU proofs, not GPU math.
- `strata-mtp-grouped-round-confidence-tests-20261014.log`:8 prior native
  confidence/packet/threshold/repair tests pass.
- `strata-mtp-grouped-round-sampler-tests-20261014.log`:10 prior native
  sampler/request ownership/payload/shader tests pass.
- `strata-mtp-grouped-round-prefix-tests-20261014.log`:every supported target
  acceptance/rejection/EOS width passes its existing pure controller check.
- `strata-mtp-grouped-round-build-20261014.log`:core release build passes.
- `strata-mtp-grouped-round-shaders-20261014.log`:317 ordinary shader bodies,
  122 physical-pointer and32 cooperative-matrix variants validate offline with
  the existing pinned2026.1 validator. No GPU/device initialized.
- `strata-mtp-grouped-round-{core,off}-clippy-20261014.log`:strict core/off
  Clippy pass. Formatting/whitespace pass. Existing duplicate-bin-path Cargo
  notice remains. No full-suite claim or waiver of prior228 failures.

No GPU/model/device, live correctness/recovery or speed trial, sudo, service,
admission/residency, public API, Vashti, installation, deployment or release
operation. Canonical checkout remains clean; inherited isolated bytecode remains
untracked/preserved. The user retains BOTH GPUs; obtain availability before any
GPU/model/residency/admission work. CPU source work may continue without sudo.
Whole rewrite remains incomplete and goal active. Next: original device residual
handoff plus captured round/step/chain dependencies, then remaining full-source,
runtime and protected correctness/recovery gates before speed trials.


## Original verifier device residual grant for accepted MTP rounds (2026-10-14)

Source `2532963`, after `f671543`, isolated `codex/strata-rust-port`. Continuing
accepted MTP rounds now consume the original verifier residual owner directly.
This supersedes that part of the previous grouped-round handoff; **host packet
readback, initial/final host-carry paths and full captured scheduling remain open**.
No speed-based path selection or live model execution was performed.

### Connected source and lifetime

Pinned Intel `Verifier::final_R_all()` returns the actual device R_ carved by
`verify.cpp:507/544`. Serial `generate.cpp` binds it directly to MTP; native
`capture_round` copies its accepted rows into Rin_. Rust's packet producer already
receives the actual `outer.residual[current]` BufferId. `resident_target_rows.rs`
now mints a private `DeviceCarryReceipt` only after its real compact packet has
passed decode/finite/fault checks. It records that exact source ID, opaque graph
binding, target write generation, original start/count and bounded wiping picks.
No raw handle, erase/free authority, Clone/Debug/serialization or persistent
content is exposed by the receipt. The original host output lease still must be
dropped before target prefix commit.

`ResidentModel::borrow_target_carry` checks the completed original projection/
state boundary, current generation/binding, actual committed accepted cursor,
source membership in the original outer residual pair, complete owner extent
and physical provenance. BufferId authenticates arena/reset epoch/slot, including
multiple instances sharing one public graph. Receipt generation is invalidated
before **all native chunk purposes, prompt prefill, scalar stepping and warm
reset** may reuse/erase target scratch; overflow rejects without wrapping. Target
state-only prefix replay and public expert adaptation do not overwrite that R
owner, so a receipt from the real verified window remains valid across commit.

The resulting `DeviceCarry` holds an immutable lifetime-bound borrow of the
actual parent arena/receipt and original retirement flag. Safe caller code cannot
mutate/reset/close the target while it exists. It validates the supplied original
prediction prefix and permits only the accepted row interval. Copies authenticate
same client/stream, full source/destination/fault owners and actual disjoint
physical extents before borrowing read-only source views. No sliced view becomes
an owner, no extra GPU source allocation is made, and no source is erased by MTP.

The actual selected global controller now retains this receipt instead of a
host residual vector between rounds. Next native MTP entry copies the selected
last R row directly into the existing B::Carry and the complete accepted R prefix
into the existing front CARRY owner. Original token staging, front clearing,
rewind, last-row R/injection/mixed selection, Q8_1 preparation, real rest/head,
confidence and exact target verification all remain connected. Both zero-threshold
queued chains and positive-threshold serial chains retain the source borrow
through their actual final result/sticky-fault fence. Only that completed return
clears its in-flight marker, at one documented unsafe proof site. Dropping an
unjoined grant retires the **original source context** and leaves original parent
cleanup authority intact. Enqueue is never credited as source completion.

The implementation uses the existing guarded CubeCL word-copy kernel and native
queue dependency tracking. No new WGSL, MAIN root or runtime envelope is added.
Core target HOST_INCREMENT explicitly adds8192 bytes for receipt/grant/pick and
resource metadata:12,582,912 ->12,591,104. Legacy feature-off quotes remain
unchanged. Full private source/destination padding stays under original owners'
guards/erasure. Receipt coordinates, source ID, generation and pick storage wipe;
no request values enter logs, shader source or cache keys.

### Remaining fidelity and acceptance work

The existing full private verifier packet still contains host residuals, and the
controller still decodes them for its initial/final legacy carry/repair paths.
This block removes **their upload as the continuing accepted MTP source**, not
all residual readback. Complete the device source through first proposal and
final/count1 cache repair, then a finite-checked compact ids-only native result
path; do not silently weaken full source fault/finite validation to save readback.
The existing host laboratory interfaces can remain separately scoped.

Physical original-source placement/numerical copy, all-prefix GDN/QSA/PLE state,
queue ordering, unknown-completion retention, erasure and recovery are unverified.
Stable complete native round/step/branch captures, mapped/event publication,
applicable nonblocking/force-token chains and commit overlap remain incomplete.
Runtime input/context/cache/ring sizing and the complete source audit are still
required. Neither narrow CPU tests nor an incidental timing number opens the
whole-engine throughput gate.

### Final CPU receipt

Under `/media/jmicko/faststore/kuno-development`:

- `strata-device-carry-tests-20261014.log`:4 new pure checks pass: every accepted
  prefix against generation/binding/cursor, rejected/changed prediction prefix,
  full-owner prefix/last-row interval math and unknown-completion retirement
  without clearing prior poison. Generic CPU identity fixtures create no opaque
  graph binding, arena, GPU source or ownership grant.
- `strata-device-carry-grouped-tests-20261014.log`:4 grouped-front/window/entry
  checks pass. The entry test uses the actual shared pure validator, not a forged
  device receipt. `strata-device-carry-confidence-tests-20261014.log`:8 pass;
  `strata-device-carry-sampler-tests-20261014.log`:10 pass.
- `strata-device-carry-prefix-tests-20261014.log`:the all-width acceptance/
  rejection/EOS check passes. `strata-device-carry-packet-tests-20261014.log`:
  original core-eight compact packet/padding/finite check passes.28 total focused
  checks. No full-suite run or waiver of the previous228 failures.
- `strata-device-carry-build-20261014.log`:core release build passes.
- `strata-device-carry-shaders-20261014.log`:317 ordinary shader bodies,
  122 physical-pointer and32 cooperative-matrix variants validate offline using
  the pinned2026.1 validator, without device initialization.
- `strata-device-carry-{core,off}-clippy-20261014.log`:strict core/off Clippy pass.
  Formatting/whitespace pass. Existing duplicate binary-path notice remains.

No GPU/model/device, speed trial, sudo, service/admission/residency, public API,
Vashti, installation, deployment or release operation. Canonical checkout remains
clean and inherited isolated bytecode remains preserved. BOTH GPUs belong to the
user; reconfirm before GPU work. Source/CPU work may continue. Whole goal active,
full rewrite incomplete. Next connect remaining direct-carry/compact results and
native captured-branch dependencies before protected correctness/recovery gates.


## Native target ids-only results and first/request-exit device paths (2026-10-14)

Source `9edc3f6`, after `2532963`, isolated `codex/strata-rust-port`.
**Source/CPU progress only. The current core speculative harness still creates
an incompatible separate MTP client; it cannot execute the new shared-client
path until the construction/accounting/cleanup block below is implemented.**
This explicitly supersedes any implication that call-site connection alone
established a runnable complete engine. No live speed gate is open.

### Selected target dataflow and private packet

The global controller explicitly calls `Scope::use_device_residuals()` before
its last normal prompt decode. The lexical request flag defaults false for old
laboratory users, validates idle phase, client and original graph at packet read,
and selects the device-source output for **every** target result in this
controller. It is not a benchmark-driven optional rollback. No gateway/API or
Vashti protocol is changed.

The selected packet producer still consumes the real full-vocabulary sampled/
greedy picks, original complete residual root, sticky fault and full output root.
`strata-device-packet.wgsl` uses one128-thread group per compiled row. It scans
all10240 active residual values for finite bits, joins a workgroup-local atomic
bad flag with barriers, and publishes a separate status per row. The header
writer never races another group's newly discovered fault. CPU decode rejects
any row status, original sticky fault, invalid id, nonzero inactive id, bad
geometry/version/reserved word or wrong packet length. Successful payloads own
wiping ids and coordinates; their host logits/hidden vectors are empty and
`hidden()` is unavailable. The original source receipt is minted from that
finite-checked result and still authenticates the actual arena BufferId.

Internal packet version3 (not an application wire ABI):

| Word | Meaning |
| --- | --- |
| 0..8 | version3, active count, original start, vocabulary248320, HC width10240, compiled capacity8, original sticky fault, reserved0 |
| 8..16 | active selected ids (<248320), inactive0 |
| 16..24 | per-row residual finite statuses, all0 for acceptance |

Logical readback is96 bytes rather than327744 for the old core-eight carry
packet. This is a source byte count, not measured PCIe traffic or speed.
The complete8MiB packet owner/clear/erasure authority and original private read
scope remain; the96-byte view does not acquire owner authority. Legacy packet
version2 remains for callers without the explicit device-source policy. No new
MAIN root, runtime quote or host reservation is added by this block; the existing
device-grant8192-byte host increment remains.

### First cell, count1 windows and native request exit

Pinned `mtp.cpp::draft_first` (1954) stages the actual token/R pair in row0 and
calls draft with accepted row0. Default `mtp_catchup_all=false` truncates its
front to that one row; the unused duplicate source staging is not reproduced.
Rust now seeds its pending device receipt from the actual final-prompt output.
The first source's count1/untouched teacher cursor is distinguished from an
accepted catch-up requiring completed cache authority, including cell0 for a
one-token prompt. No host target residual is extracted/uploaded by the selected
controller for that first proposal or later accepted proposals.

When the controller supplies only one target input, the device front still
appends its actual current token/R cell without running a draft rest/head.
A new distinct `LastFront` capability records only a completed KV front. It is
not interchangeable with `LastForward` and cannot authorize `follow_draft`.
Normal forward publication, step, append, rewind, prompt teaching, failure and
close invalidate the appropriate old stamps. Both device-source copies and the
actual private fault fence complete before front position/capability publication.

The source audit of `generate.cpp:12310..12345` found that native EOS/output-cap
exit commits the target prefix and performs **no further MTP catch-up**. The
selected Rust exit now follows that rule instead of retaining the old eager
final repair. `finish_native_request` checks completed scope/live-output state
and discards provisional follow/front permits. It does not mark cache contents
accepted, erase buffers or claim memory was released. The actual original owners
retain all provisional KV and perform full erasure/closure: existing harness
`target.close()` around2470 and `draft.close()` around2537, with their audits
before reporting success. No live erasure/recovery claim is made here.

### Newly verified construction gap: next required source block

`daslab_language_probe.rs::draft_client` (910 onward) still requests another
`wgpu::Device`, calls `init_device`, and proves a **distinct** runtime client,
even for the same Intel physical adapter. Its caller (1125 onward) configures
independent transient I/O and rejects inherited target request/backlog policy.
`ResidentMtp::construct` (around1390) additionally requires a completely fresh
client and starts its6GiB `MainAdmission` at baseline0. These are old isolated
laboratory assumptions, not a valid shared-runtime implementation of Strata.

The new source grants, existing prompt device teaching and coupled sampler all
correctly require the **same** actual client/device context. Consequently the
current core speculative entry cannot pass those preflights. Do not bypass or
weaken them, substitute a same-physical-adapter comparison, transfer foreign raw
handles, or claim completion from these CPU checks.

Next unify the selected Intel constructor under the actual target Vulkan device/
client, with truthful combined pre-device MAIN/runtime/host reservations,
allocation attribution/fences and context retirement. Reconcile fresh-client/
per-owner ledger assumptions against `main_admission.rs`, target admission and
both real arenas. Audit shared transient-I/O policy and final close ordering;
merely cloning the client or treating existing peer allocations as zero would
be incorrect. Keep non-core independent legacy laboratories separately scoped.
This work can proceed on source/CPU without sudo or GPU access.

Then complete stable native branch recording/warm-up and mapped/event/chain/
commit overlap dependencies. Two additional remaining source audits are explicit:
current draft count is capped by remaining outputs/count1 front-only, whereas
native draft uses configured max_drafts; reconcile native window/no-MTP/suffix
policy as one block. Positive-threshold MTP currently still uses its old scalar
`DraftOutput` readback containing an unused hidden vector; the **target** ids-only
change does not fix that draft observation path. Arbitrary input/context/cache/
ring policy and the entire dependency checklist/physical correctness and recovery
remain required before any throughput trial.

### Exact final CPU-only validation

Logs under `/media/jmicko/faststore/kuno-development`, prefix
`strata-device-ends-accepted-`, suffix `-20261014.log`:

- `native_device_end`:4 new checks pass: initial/accepted boundary distinction,
  typed front stamp binding/position, all-width ids/status packet rejection and
  new producer Naga/SPIR-V lowering. No device/source grant is fabricated.
- `native_device_carry`:4 pass; `native_grouped_round`:4;
  `native_confidence`:8; `native_sampl`:10; `global_prefix_acceptance`:1;
  `core_eight_compact`:1.32 focused checks, not full-suite or numerical acceptance.
- `build`:core release build passes. `core-clippy`/`off-clippy`:strict core/off
  Clippy pass; final ownership spelling is the lint-clean last-use move.
- `shaders`:317 ordinary bodies plus the controllers (including the new ids-only
  producer),122 physical-pointer and32 cooperative-matrix validations pass with
  the pinned2026.1 validator. The standalone public-source-only checker
  `/tmp/kuno-native-device-packet-check-20261014` also emits SPIR-V accepted by
  `spirv-val --target-env vulkan1.3`. Formatting/whitespace pass.

Earlier final-repair staging/validation receipts are superseded by this native
request-exit source; the change follows upstream source, not any speed result.
The omitted prompt-teaching front-stamp invalidation and final Clippy spelling
were repaired before the above exact final checks. Existing duplicate-bin-path
Cargo notice remains. Prior228 full-core failures are not waived or audited by
these focused tests. No GPU/model/device, speed trial, sudo, service/admission/
residency, install, release or application/API operation. Canonical clean checkout
and inherited bytecode preserved. Both GPUs remain with the user; obtain
availability before GPU work. Goal active, rewrite incomplete. Resume with the
shared-context construction/accounting/cleanup block, before captured execution.


## Shared native Intel construction, accounting and close (2026-10-14)

Source `e418929`, isolated `codex/strata-rust-port`, after
`9edc3f6`. This supersedes the separate-client constructor blocker in the
preceding receipt. It is source/CPU acceptance, **not a runnable whole-engine or
physical-recovery claim**. No GPU/model or speed run occurred; both GPUs remain
with the user.

The selected `daslab_language_probe::run` skips `draft_client` in the core.
Before target construction, its executor reserves the original Q2 MTP public
plan, full64MiB private envelope and128MiB public-source envelope. After the
actual target/global source owners are constructed, a non-clone one-shot
`NativeMtpAdmission` permits `ResidentMtp::new_shared_strata_q2` to consume the
same original graph binding, client and context-retirement flag. It does not
accept a same-physical-adapter comparison as client identity. The existing
device-carry/prompt-teaching/coupled-sampling identity checks remain intact.
No foreign handles, duplicate Vulkan device or separate6GiB MTP ledger is used
by this core path. The independent legacy laboratories retain their old path.

Both real owners now precharge and observe the same `Arc<Mutex<MainAdmission>>`.
The MTP ledger adapter delegates to that actual target account. Actual target
baseline bytes (including the original shared IQ table) are retained; they never
become a fabricated zero. Allocation failure/drop grants no refund, retirement
and sequence failure remain sticky, and only completed fences reduce debt.
MAIN page admission now follows the actual allocator's core512MiB/legacy128MiB
upper bin; the former128MiB hard stop made the three512MiB prompt roots and
native persistent cache chunks inadmissible.

The fresh protected Xe proof still checks actual Intel/discrete/Vulkan identity,
fresh runtime, matching device and nonzero/unmasked consistent occupancy. Core
physical preflight reserves one31GiB combined envelope plus512MiB driver margin.
The old uniform-target-plus-independent-draft arithmetic is no longer applied
to this topology. Before allocation, the actual fixed target/private/vector/
source/baseline/MTP owners,49 original address-table roots and combined runtime
window leave the remaining expert byte budget. The global ranked-prefix loader
consumes that exact budget; its actual requests must fit it plus their separately
priced tables. HOST complement geometry is recomputed from that same prefix.
The table charge is49*32768=1,605,632 bytes; it is not hidden in expert capacity.
The shared runtime window is256MiB, selected by the isolated core's vendored
`kuno-strata-shared` feature; legacy fixed target I/O remains128MiB. Target and
MTP runtime quotes are added and checked together. Per-buffer bounds and private
owner erasure are not enlarged by the runtime-window change.

MTP observations now join the actual shared request-I/O nonce/checkpoint rather
than sanitizing an unrelated stream. The original MTP arena closes/inspects all
its roots before target close poisons the shared account. Its own close may
truthfully observe still-live target bytes; it cannot claim an empty allocator.
The target, executor/global sources and shared public IQ table then close.
Only a final real sync, full transient-I/O audit, cleanup/sync and allocator
sample with all three counters zero can publish final recovery. Every failure
retires the original context; this final observation neither revives nor credits
the poisoned admission ledger. The target-only core also takes that final audit.
The profile close validator now includes all25 native-front roots and native
split-attention geometry, matching the original arena's full audit. Core MTP
private geometry is69 roots /18,776,064 bytes; this numeric plan is not itself
proof of erasure.

Final CPU-only receipt (logs under
`/media/jmicko/faststore/kuno-development/strata-shared-context-accepted-*-20261014.log`):

- `core-native_shared`:7 checks pass (shared capacity/debt, prompt roots,
  physical-plan rejection, final recovery guards, complete MTP receipt and
  global table/persistent-root pricing). `core-main_admission-tests`:10 pass.
- `core-native_device_end`:4, `core-native_device_carry`:4,
  `core-native_grouped_round`:4, `core-native_confidence`:8,
  `core-native_sampl`:10, `core-global_prefix_acceptance`:1 and
  `core-core_eight_compact`:1 pass. Including the3 overlapping MAIN cases,
  this is46 distinct core checks, not full-suite or numerical acceptance.
- `off-native_shared`:7 pass for `strata-rust-port` without the core feature.
  `core-clippy`/`off-clippy`:strict binary Clippy with `-D warnings` pass.
  `core-build`:release build passes. Root formatting, vendored-file formatting
  and whitespace checks pass; the existing duplicate-bin-path Cargo notice remains.
- `shaders`:317 ordinary shader bodies, the controller producers,
  122 physical-pointer cases and32 cooperative-matrix cases pass offline with
  the pinned2026.1 validator. The early main branch returns before device/model
  initialization. No numerical or physical execution proof is implied.

The vendored library's own policy-unit target was not run: Cargo rejected
feature selection for that dependency outside the workspace. Its source policy
is compiled by both accepted application configurations, and physical shared-I/O
configuration acceptance remains pending. No full-suite waiver is granted.

Remaining fidelity/runtime gates are explicit: target embedding/output weights
are still uploaded twice rather than sharing pinned target dense roots; merely
cloning cache handles while letting target LRU eviction refund them would be
incorrect. Concrete next source audit: `GpuExecutor::weight_inner` releases LRU
entries and keys them by tensor/expert/first; its hit path does not check a new
request's row extent. `ResidentMtp::load_page`/`tile_plan` independently upload
the original target tensors. Shared owners must establish common tile geometry,
original dtype/graph/full-range identity and pinned lifetime for both consumers,
price roots once, and release only after both owners complete. Do not turn a
matching cache key or cloned handle into a shared-weight ownership proof.
The public request fixture and protected host/cgroup envelope need
whole-model acceptance with the newly computed HOST complement. Stable native
branch/round capture and warm-up, mapped/event/chain/commit dependencies,
positive-threshold draft hidden readback, native draft-count/remaining-output
policy, arbitrary prompt/context/cache/ring behavior and the full dependency
checklist remain incomplete. Historical228 full-core failures remain unaudited;
focused tests do not waive them. No speed trial before the whole-source and
correctness gates. No installed service, API, Vashti, release or driver change.


## Shared original dense roots and GPU embedding (2026-10-14)

Source `9de7fef`, after `e418929`, isolated `codex/strata-rust-port`.
This supersedes the duplicate target/MTP embedding/head ownership gap. Source
acceptance alone is not whole-engine, numerical or physical-recovery acceptance.

Pinned Intel `mtp.cpp::bind/record_front`, `verify.cpp` and `generate.cpp` load
the target embedding/head before expert-cache construction, then borrow those
same roots in MTP. `strata_shared_weights::Weights` now pins the original
IQ4_XS embedding and Q6_K head for that actual graph and client, with common
16384-row tiles (final2560). The32 exact4096-aligned persistent roots occupy
859,189,248 bytes. Their original payloads, all half scales, zero padding and
physical extents are checked. The one shared account charges them inside the
existing5GiB static target envelope; the MTP peer quote subtracts these roots
and retains all of its additional public/private/source owners. The target LRU
can only use the remaining static envelope. A separate64MiB source envelope
bounds shared-weight startup staging, and every upload joins before the next.
No derived weight file or model mirror is saved.

The target head bypasses ordinary LRU through exact original-role/row/extent
lookup. MTP borrows the same handles and retains their explicit owner. MTP close
drops these views/owner before target closure; target refuses closure while
another explicit owner remains. No eviction/refund grants lifetime or MAIN
credit. `CloseReceipt.public_bytes` means closed addressable public spans,
including shared views; it does not falsely claim those target-pinned roots
were physically freed. Final real shared-context recovery audit remains required.

The selected native target MAX8 decode and native prompt paths now gather
original embedding rows on GPU and broadcast into all four HC streams. Existing
private token/fault owners are used. Actual arena membership or authenticated
prompt-stage loan bounds, disjoint logical spans and unchanged graph/client
identity are required; a view never gains erase/free authority. Original owner
erasure remains unchanged. The core controller also queries the same immutable
actual backlog policy at entry and exit, repairing an old mismatched final
legacy-stream receipt. Legacy paths retain their separate behavior.

Explicit Vulkan substitution: the original public embedding/head are split
into16 bounded tiles each; gather dispatches visit these tiles instead of
Strata's contiguous whole-table allocation. Storage format, GPU placement and
consumers are preserved; physical scheduling and numerical equivalence remain
unverified. Optional native reduced-draft-head policy still requires audit.

Latest user authorization returns both GPUs for protected correctness work and
refreshes sudo. It supersedes older notes reserving GPUs for the user. No GPU
work or throughput trial has been performed in this block. Continue the complete
rewrite: compact positive-threshold observations, native capture/warm/event/
chain/commit dependencies, configured draft policy and arbitrary-input/runtime
gates. Recheck protected HOST/cgroup envelopes for actual complementary bytes.
Prior228 full-core failures remain unaudited. Installed services, APIs, Vashti,
drivers, releases and original artifacts are unchanged.

Validation for source9de7fef (CPU only), log prefix
`/media/jmicko/faststore/kuno-development/strata-shared-weights-accepted-`:
core `native_shared`11, `native_device_end`4, `native_device_carry`4,
`native_grouped_round`4, `native_confidence`8, `native_sampl`10,
`global_prefix_acceptance`1 and `core_eight_compact`1 checks pass. Feature-off
`native_shared`11 checks pass. Strict core/off binary Clippy and core release
build pass. Offline shader set318 ordinary bodies plus controller,122
physical-pointer and32 cooperative-matrix cases pass; no device is initialized.
Formatting/whitespace pass. Existing duplicate-bin-path Cargo notice remains.
These focused checks are not a full-suite waiver or whole-model acceptance.


## Native draft observations, lookup policy and exact commits (2026-10-14)

Source `a26adc2`, after `9de7fef`. Connected source/CPU acceptance; whole
rewrite remains incomplete and no speed trial is authorized. Pinned references
are Intel `mtp.cpp::draft/capture_round/capture_step`, native `generate.cpp`
speculation and `src/spec/{suffix_drafter,draft_policy}.cpp`.

Positive confidence thresholds previously called `draft_output` per cell,
copying its10240-value hidden vector to host although the next cell used the
original device owner. Both threshold modes now use one exclusive queued
window: the same full private packet contains only token ids, probabilities
and sealed sticky fault. A positive threshold joins/validates that original
packet after each cell and stops before the next cell on probability<minimum;
zero threshold joins once at the chain end. Both keep Residual2 on GPU and
feed Carry/Token through the existing same-stream original-source copy. Only
the completed final cell grants LastForward authority. Device source grants
stay borrowed through the actual final result/fault fence. Cancellation,
invalid padding/fault or incomplete completion still fails the whole window.

Explicit remaining Vulkan substitution: logical packet16 words / full32KiB
owner, with actual fenced readback and complete padding validation; this is
not Strata's permanently mapped tiny per-step observation/event publication.
The scalar/legacy diagnostic APIs still expose their explicit hidden outputs;
the selected native controller no longer calls those APIs. Source compilation
does not establish device numerical/state equivalence.

`strata_speculation` ports the native four-way trigram hash index, including
the current occurrence consuming one of the four entries. Longest64-token
match wins, most recent wins ties; proposals after provisional drafts do not
index those drafts, and the temporary tail is wiped before truncation. The
fixed8192-token history and16384-slot wiping index plus bounded proposals fit
inside the existing4MiB request HOST quote; no token-dependent graph key,
output log or persistent history is introduced.

The selected controller now uses the pinned eight-size cost shape, per-match
lookup/chain priors, cost and acceptance EMAs, three unmeasured-size probes,
300-round stale-cost refresh, and exact reached-chain accounting. Default
native config enables suffix minimum3 and leaves optional lookup-chain off;
its actual request config carries target/MTP windows and chain parameters.
Whole-model adapter is still the fixed public fixture, not an application API.
Cost observations span this verify/commit plus the next actual draft, before
choosing the following round's policy; no timing is published or used to
reject coupled port changes. Optional chained suffixes use the real selected
MTP probability product and only target-verified emitted tokens reach history.

Configured draft windows no longer shrink to remaining requested output.
Target commits the entire exact-match input prefix; EOS/output cap limit
delivery separately, matching native generate.cpp. Terminal state can
therefore be ahead of the last emitted token, and final acceptance checks
that real committed cursor rather than inventing another draft/catch-up.
All continuing rounds still require emitted prefix=committed prefix. Context
exhaustion rejects the full configured window rather than silently clipping.

CPU receipts under `kuno-development/strata-native-policy-*-20261014.log`:
6 new translated native policy/suffix/commit cases,8 native confidence cases,
4 grouped-round cases and1 prefix-acceptance case pass. Core/off strict binary
Clippy and core release build pass. No shader producer changed in this block;
confidence packet producer lowering is included in those confidence checks.
Frozen protected artifact sourcea26adc2:SHA256
`1060d768291811596c56e022bc556e0308143dbec46073642a9d800db0e4fd83`.

A first protected code/Q2 completion check stopped before model loading on
`native model chunk rejected at source line1155`. This exposed a REAL stale
constructor expectation: the original target root sum still assumed F32 QSA
cache/pool roots. Independent CPU geometry confirmed590,151,680 native base
bytes instead of785,186,816, plus the unchanged GDN replay roots. Source
`575971f` prices all12 native QSA full owners together: each32MiB cache becomes
16MiB and each512KiB pooled-key root becomes1MiB. Exact sum/root-count equality
checks remain; no relaxed limit or fallback was added. The owning plan test
now includes actual selected GDN replay inputs, full native padding, exact
constructor aggregate and old feature-off behavior. Follow-up execution pending.

Failed-run evidence:
`strata-native-policy-a26adc2-check-20261014` and
`strata-native-policy-a26adc2-root-run-20261014.log`. Cleanup accepted: actual
Intel used bytes50,688,000, no NVIDIA compute process, all classified GPU fault
counters unchanged0, normal admission restored, Kuno/Ollama/Vashti HTTP200.
No reset/reboot, driver, installed binary, API/UI or release change. This failed
check is not numerical/whole-engine acceptance or a speed result.

Capture audit next: `run_global`/`strata_global_cpu::prepare` still use actual
per-layer packet/input readbacks; parent flush enters native `Graph::execute`,
which waits its completion fence. Exact-node recording uses a16-entry LRU,
not native warm capture of all stable target windows and MTP round/step
branches. Complete the mapped doorbell/input/output ownership, original
CPU/GPU plan publication, branch parameter staging and event/commit overlap
as a connected block. Merely raising an LRU size or keeping buffers without
those consumers does not finish capture. The current adaptive exchange IS
called from LatestTargetRound::finish; its synchronous publication/overlap
still needs the same whole-source audit. Prior228 full-core failures remain
untriaged beyond the explicitly reproduced/fixed constructor issue.

Further connected constructor audit: source395f4b2 joins the actual native
QSA MAX8 geometry (22,052,864 vs17,825,792 bytes) into BOTH the component check
and aggregate NEW_MAIN. Source1554d5c also joins both audited copies of the
native GDN compatibility INFO into the parent expected value and real runtime
quote:5,013,504 vs4,718,592 bytes. These were missing connected consumers of
previously changed family layouts, not reasons to abandon those families.
The exact old-owner test passes at source575971f; QSA full-page inspection
confirmed its original scalar owner also includes the extra512KiB pooled-key
root (20,054,016 bytes per native QSA stage), and its stale CPU test expectation
needs correction. The broader core suite remains unaccepted.

Protected attempts at source575971f and395f4b2 stopped before payload load at
parent layout checks1317 (QSA batch) and1702 (GDN INFO), respectively. Both
recovered the exact physical50,688,000-byte Intel baseline, no NVIDIA compute,
unchanged zero classified faults and normal admission/HTTP200. Logs/output
directories: `strata-native-qsa-geometry-{root-run,check}-20261014` and
`strata-native-qsa-batch-{root-run,check}-20261014` (root-run files end.log).
Source1554d5c follow-up pending. No speed number or full correctness acceptance.

The actual protected public fixture now selects native spec4/min-probability0.5
instead of treating MAX8 compilation as the request default. The saved upstream
comparison configuration independently confirms those settings. Full-vocabulary
drafting, FP16 KV and the4096-token owned context remain the explicitly selected
source-port mode; the earlier upstream comparison used its CJK subset, INT8 KV
and32768 context. Do not claim matched whole-engine performance from either mode
without the explicit final completeness/substitution and numerical/state audit.

The owning scalar plan check passes at575971f. Source3fe048e reconciles the
independently observed20,054,016-byte native QSA stage and2,506,752/5,013,504-byte
GDN INFO expectations. Final focused CPU checks pass:2 exact-full-page cases
(QSA and PLE),1 GDN INFO inventory and1 original model-owner case. Sourcebb705ae
strict core/off Clippy and core release build pass. Existing duplicate-bin-path
Cargo warning remains. No all-suite waiver is implied.

The source1554d5c protected check passed the parent geometry gates but stopped
with `resident runtime I/O cleanup failed`; target/draft owning cleanup did NOT
produce an acceptance receipt. Process-exit physical recovery still passed
Intel50,688,000 bytes, no NVIDIA compute, unchanged zero faults and restored
admission/HTTP. Treat physical process recovery and private owning erasure as
separate evidence. Sourcebb705ae adds fixed-public backend source-location
diagnostics; the same failure occurs without invoking that backend fault helper.
Sourceab4bf67 adds static constructor phase and cleanup-retirement booleans to
identify the boundary. Neither diagnostic prints backend descriptions, request
values, token ids, INFO buffers or images. Follow-up pending; no speed trial.

Sourceab4bf67's protected trace isolated the failure AFTER complete target
construction, at `phase=shared_target_weights`. No vendor failure occurred:
the native public upload uses no private transient stages, so its actual
completed audit returned all-zero I/O counts. The old private-readback validator
incorrectly demanded a nonempty private audit for this public observation and
retired the shared context. This is a parent-contract bug, not a GPU crash.

Source172e6df gives only `ResidentAdmission::shared_observation_checkpoint`
a separate completed-empty-audit policy. It still invokes the real same-stream
full sanitizer/audit, requires a nonzero completed epoch for an empty receipt,
exact byte accounting and empty-buffer/count consistency, joins original
ordered sources, checks context/sequence health and obtains actual post-fence
MAIN telemetry. It does not reacquire the MAIN mutex from inside that mutex.
Active request-I/O scopes retain their original reuse/checkpoint branch.
`readback_audited` and the ordinary private-readback validator STILL reject
empty receipts. No synthetic receipt, skipped drain, retirement reset, extra
memory credit or ignored cleanup failure is introduced. Failure cases retire
exactly as before. The focused CPU test covers empty/private rejection,
missing epoch, inconsistent bytes/buffers, overflow, nonempty acceptance,
audit errors and prior retirement. Build/protected retest pending.

Sourcebb705ae/ab4bf67 fixed-public diagnostics print only static constructor
phases, cleanup booleans and source locations. They do not dump error
descriptions or request bytes. Both protected diagnostic attempts also
recovered the exact Intel50,688,000-byte baseline and normal services while
properly reporting that owning private erasure was not accepted.

Source172e6df strict core Clippy/build and its completed-empty-audit CPU check
pass. The existing production-audit negative test also passes unchanged
(`strata-native-empty-public-private-audit-exact-20261014.log`); the earlier
`private-audit` filter selected0 tests and is NOT coverage.

Protected source172e6df now successfully constructs target AND uploads all
shared target weights. It reaches `phase=shared_draft` and fails the historical
`MTP original resident kernel limits` check. Source560474b selects a native
Q2-specific validator: the real split-attention native dispatches use the
common1024-thread/18-storage-binding core and65535-grid/page-root capabilities,
not the legacy flat QH*CONTEXT dimension. The old validator remains unchanged
for actual legacy/self-contained consumers. Native source validation remains
strict; no hardware limit is raised or fabricated. A CPU fixture tests native
acceptance at65535 and smaller workgroup/grid/binding/root rejection, alongside
continued legacy-grid refusal. Shared-context cleanup retires on constructor
failure as before; physical process recovery again accepted exact50,688,000
Intel bytes, no NVIDIA compute, zero unchanged GPU faults and normal services.
Build/test/protected source560474b follow-up pending.


## Shared construction continuation and real CPU/GPU submit boundary (2026-10-14)

This receipt supersedes pending construction notes above; source through
`dbf2312` on isolated `codex/strata-rust-port`. Whole rewrite remains incomplete.
No speed/profile trial, installation, API/UI/release change or GPU reset.

- `172e6df`: completed empty public-upload audit accepted only at the shared
  observation checkpoint; ordinary private readback remains nonempty/strict.
  New empty-audit unit and the original exact private-audit refusal case pass.
- `560474b`: validate the actual native split-attention grid/capabilities,
  preserving the old legacy flat-grid check. Thirteen native_shared tests,
  strict core/off Clippy and frozen core release pass.
- `5b91d9a`: query all137 original MTP owners in batches of at most128, preserving
  the backend limit and the complete cross-batch physical-span/alias proof.
  Fourteen native_shared tests, strict core/off Clippy and frozen release pass.
- `851475a`: source loading requires untouched construction, not an accepted
  token. The new initial boundary proves the same model/graph/client/arena,
  fault/retirement/source health plus PLE next=position=generation=0, no pending
  operation or selected runtime PLE policy, unseeded token history and zero
  convolution history. Request geometry/cursor/one-token sampling may inspect
  this state. It cannot mint CompletedNativeProjectionBoundary; lab policy
  switches still require an actual accepted nonzero cursor. Fifteen native_shared
  tests and strict core/off Clippy pass. Full unit suite still unaccepted.
- `f597084`: connect a distinct submit-only server/client/stream path to the
  real core expert/shared work inside the existing scoped CPU pool. Previously
  parent.flush claimed overlap while native Graph::execute waited its fence.
  The new path keeps the recording, full node/resource inventory, actual queue
  plan and fence alive. Only successful completed join credits execution and
  clears nodes. Partial submission/failed join remains terminal. Later compute,
  transfer, ordinary flush, readback/sync/sanitizer joins before reuse; submit-only
  skips allocator/uniform cleanup. Other servers reject the extension; ordinary
  flush keeps its old synchronous behavior. Frozen core release passes; focused
  checks/strict Clippy and protected follow-up are pending at writing.
  This does NOT implement mapped CPU doorbells or stable whole-window capture.

Protected frozen checks: `strata-native-mtp-limits-*`,
`strata-native-metadata-batches-*`, `strata-native-initial-boundary-*`, under
`/media/jmicko/faststore/kuno-development`. The560474b check passed model/draft
load then rejected137 metadata bindings. The5b91d9a check passed both models
and reached global_sources, rejecting the old zero-cursor PLE check. The851475a
check passes that boundary, then fails `resident graph device operation` during
original global source preparation. There is no accepted full source-load,
whole-model completion, numerical/state or owning-erasure receipt yet. All
three process exits restore exact Intel50,688,000 bytes, no NVIDIA compute,
zero unchanged classified GPU faults, normal admission and all three HTTP200.
Failed owning cleanup is not retroactively accepted because physical exit worked.

Next: locate the global-source preparation error, then finish mapped observation/
CPU doorbell plan service, stable complete target/MTP branch capture/warm-up,
parameter staging and commit/event overlap as one connected schedule. Finish
arbitrary-input/runtime and complete numerical/state/recovery gates before speed
trials. Retain coupled source changes regardless of incidental duration. Do not
waive historical228 full-core failures; only the specifically repaired cases
have new acceptance evidence. Both GPUs authorized; sudo refreshed in Demons.


Source `dbf2312` adds bounded, fixed-public global-cache stage/error CLASS
classification (no backend strings/content); it includes f597084 and builds.
Frozen SHA256 d9a465e70d61d7e90046a07fb26b7b6a787130f992dde3d9016d4013a5bbf7a4,
`strata-native-global-source-trace-build-20261014` in the evidence directory.
Protected completion running at writing; no accepted inference/speed claim.


The dbf2312 protected check reproduces the generic source-setup failure WITHOUT
triggering either cache-upload/completion classifier: do not describe that as
an upload failure. Source audit found a SECOND complete-inventory call in
Sources::build_tables directly sending GPU+HOST roots to the128-binding API.
Source8b5f7aa centralizes always-retained bounded inspection and connects both
that address-table builder and the GPU cache inventory. Full root extents,
physical-address map, owner identity, cross-batch uniqueness/aliases and all
source consumer limits remain unchanged; no truncation/raised backend limit.
The same CPU helper fixture already covers1/128/129/137/379/2048/4096 owners,
short batches and second-batch failure. Frozen8b5f7aa core release passes:
`strata-native-global-inventory-build-20261014`, SHA256
3c9ca431d93881f67b429b689c92e0c81f8ae95209aa3143961698cdbd6b6a66.
Protected original full-source completion check running at writing. Actual
acceptance is pending; no numerical/whole-port/speed claim. The prior dbf2312
exit again recovers exact physical baseline, unchanged faults and normal services.


The8b5f7aa protected run gets THROUGH prepare_global_sources_before_prefill and
into the request's runs closure (its public_language_failed_cleanup path is
now invoked); global construction is no longer the current blocker. It stops
with resident graph device operation during request execution before a complete
receipt. Exact error stage pending. All physical recovery/service/fault checks
pass again. Sourcea3cd02a adds fixed-public controller phases and static stream
error classes. Its build failed due to a missing LaunchError import, corrected
byea2b53c; do not select the failed artifact. No backend description/token data
is printed. Final corrected release and strict core/off Clippy pending.

Source throughdbf2312 passes15 native_shared and6 native_speculation cases;
strict core/off Clippy passes through8b5f7aa. Exact logs:
strata-native-submission-final-{shared,policy,core-clippy,off-clippy}-20261014.log.
Do not confuse those focused CPU checks with a full core regression or numerical
acceptance, nor submit-only source compilation with a live overlap receipt.


## Full-owner prompt uploads and ordered carved copies (2026-10-15)

Corrected ea2b53c release and strict core/off Clippy pass. Frozen SHA256
4126af0c785ebaf6ab5dbdc9b38fe70f46f1fd372d114bd0b24ebc9b341d49ab,
strata-native-execution-phase-fixed-build-20261014. Protected phase check gets
through source initialization, request_preflight, prompt_prefix and prompt_chunk,
then fails resident graph device operation without a queued stream error. All
physical recovery/services pass as before; owning cleanup is NOT accepted.
No speed result. Error occurs during first prompt chunk, not global setup.

Source audit identifies a deterministic invalid private API call in shared
embedding gather: write_private requires a complete unoffset owning span but
new gathering passed a token-length sliced logit-packet view. Source539f205
uses the original FULL packet root, validates its actual layout and uploads
zero-padded full-owner bytes (at most1MiB). Shader token reads keep their exact
logical spans; no private API/erasure policy is weakened. New CPU test checks
1/8/2048 tokens, complete zero padding, original vocabulary bounds and malformed/
short/oversized owner rejection. Test and strict core/off Clippy pass; frozen
539f205 SHA25628ff3a691f62acfd160205553a8bacd442030040fc862a5aad2cfad983be7b40.
It is not separately GPU-tested; the next artifact includes the coupled fixes.

The same audit found future full-owner violations at BOTH carved prompt uploads:
resident_ple_prompt's embedding rows and strata_global_moe's grouped route map.
Sourcef4400a7 connects strata_prompt::stage_private_view at both actual consumers.
The selected original prompt scope must authorize the EXACT destination view;
the original arena supplies a distinct, full, aligned scratch root on the same
stream (32KiB..1MiB). Extents, bounded payload and source/target disjointness are
checked before mutation. Each zero-padded source window uses strict full-owner
write_private; original GPU copy kernels write only the destination subrange.

PLE uses the model logit packet, whose embedding token reads are ordered earlier;
routing maps use the existing decode-map root, unused by prompt routing. No GPU
allocation or new I/O privilege/API is added. Later overwrite of the same scratch
is an ordered graph write AFTER prior GPU copy reads, using actual retained
buffers/ranges and compute/copy dependencies; no premature root release. Source
host windows are wiping buffers, at most the already owned scratch size. Source
allocations remain under original private runtime/sanitizer ownership. Target
neighbors are preserved; padding and final erasure still cover the true owner.

This is an explicit bounded-upload/GPU-copy substitution, NOT Strata's mapped
parameter staging or proof of complete capture fidelity. Never count it as a
speed gain before all completeness/correctness gates. New staging CPU test
covers full/partial windows, total byte coverage and capacity/alignment/overflow
bounds. Full original numerical/state/owning-erasure acceptance remains pending.
Final f4400a7 release/focused tests/Clippy/protected check pending at writing.


Sourceeb4ff8e completes the carved-upload caller audit for MTP teacher tokens:
resident_mtp_prompt previously passed another sliced token view to strict
write_private. It now uses the same checked staging/copy path, with MTP's
original scalar-token root as scratch and the shared target prompt view as the
GPU destination. This requires the already implemented SAME actual context/
stream and lexical prompt scope, not an alias between separate devices. No
native shared-weight or private-upload limit is relaxed. The native_shared
staging/payload suite now has17 passing cases atf4400a7; strict core/off Clippy
and final combined corrected artifact/physical acceptance pending. These source
fixes do not imply completion of mapped parameter publication/capture scheduling.


Sourceec37fad additionally queues a FULL scratch erase after the final ordered
copy reader. Staging may legitimately use bytes beyond a scalar role's logical
word; those words must be zero again before the original MTP padding guards or
next scalar operation examine that root. It does not weaken those guards or
credit queued erasure as completed cleanup. No scope, view or target neighbor is
cleared by this scratch restoration. Failed staging remains owned by the same
terminal model cleanup. eb4ff8e/f4400a7 intermediate artifacts are not selected
for GPU acceptance; use the final combined ec37fad build below once validated.


Final combined sourceec37fad frozen release passes, SHA256
 d0d79071d30de626dd173ef93b8304f2b8eacaa914592f5addcdaacbe8b829a0
(strata-native-private-staging-final-build-20261015).17 native_shared CPU cases
pass atf4400a7; strict core Clippy passes throughf4400a7, off Clippy through
 ec37fad. Final exact core Clippy running. The corresponding protected combined
whole-model completion check is running; do not infer live acceptance from the
successful compiler. Logs use strata-native-private-staging-final-*20261015.


### Final checkpoint for this block

Sourceec37fad, immutable d0d79071d30de626dd173ef93b8304f2b8eacaa914592f5addcdaacbe8b829a0.
Final exact core Clippy passes; off Clippy also passes. Focused17 shared cases
pass throughf4400a7; later changes connect MTP's same helper and restore scratch
before existing padding checks. Prior6 speculation cases pass; full core suite
and numerical/state/owning-erasure acceptance remain unaccepted.

Protected combined ec37fad run gets through original target/draft/shared roots,
global GPU/HOST initialization, request_preflight, prompt_prefix and prompt_chunk.
It now reaches an actual Wgpu parameter/upload completion poll, instead of the
known invalid sliced private-upload call. FIRST backend failure is
vendor/cubecl-wgpu/src/compute/stream.rs:1366 (`native graph upload completion`).
That poll's original error variant is currently masked. The last displayed error,
resident retained metadata lookup rejected, is a CASCADE: a later4-owner query
finds kuno_private_io_failure:mutation_abandoned after the first failure. Do NOT
loosen metadata/health/retirement to address that cascade. Native stream174 and
sanitizer3798/3822/3587 are subsequent retired-stream cleanup failures.

Exact protected evidence: strata-native-private-staging-final-{root-run.log,
check}/20261015 naming in the development directory (actual files use the full
prefix followed by -20261015; root log:
strata-native-private-staging-final-root-run-20261015.log). host-before.json /
host-after.json are original sanitized snapshots. Parent elapsed clock is about95s,
less than its120s poll deadline. From installed wgpu-types29.0.4, PollError only
has Timeout / WrongSubmissionIndex. This SUGGESTS queue-submit validation or a
wrong successful index, not a demonstrated hardware hang; the actual variant
and preceding Wgpu validation scopes still need direct evidence. Do not simply
raise deadlines, ignore poll failure or mark completion. Inspect the FIRST queue
submission and its captured typed validation error, initialization and retained
resources before any additional full-model run.

The protected worker exits. Physical recovery accepts Intel50,688,000 bytes,
no NVIDIA compute, unchanged zero classified GPU faults, normal admission and
Kuno/Ollama/Vashti HTTP200. Owning private cleanup was NOT accepted and must not
be inferred from process exit. Current source is a development checkpoint,
NOT a selected production/release artifact. Both GPUs are idle after this block.
No reset/reboot/service binary/config/driver/model mutation. Final separate
read-only state: strata-native-private-staging-final-state-20261015.json.

NEXT source work is still the complete coupled Strata schedule: mapped observation/
CPU doorbell and plan servicing, stable target/MTP branch capture/warm-up, native
parameter staging and commit/event overlap, followed by runtime/arbitrary-input
and full numerical/state/recovery audits. This block's source fixes do not satisfy
those remaining rows. No throughput/profile comparisons or speed-driven rollback.
Goal remains paused; no claim of whole rewrite completion.


## First submission validation and native root bookkeeping (2026-10-15)

Further first-failure diagnosis: source31a2e6f, frozen55dee858694fff6453eb8a39398583baf2dff8c31cbff0d61e550553ad890b94,
strata-native-submit-validation-build-20261015. Core/off strict Clippy pass.
Protected check DIRECTLY confirms WrongSubmissionIndex(requested5276,
last-successful5275) and already-ready typed Wgpu validation scope category
validation_usage. The prior timeout hypothesis is not the failure. The failed
flush observer polls only already-ready error futures, preserves pending ones,
records typed failures before mutation-abandoned and logs static categories;
no backend description/content, completion credit, wait or health revival.
Physical baseline/fault/service recovery passes again. No numerical acceptance.

Source audit finds native register_pipeline's Wgpu initialization stub binding
ALL original inference views to a zero-work dispatch of the original pipeline.
Distinct carved read/write regions can share the same physical VkBuffer.
WebGPU usage validation treats the buffer as one scope even though the actual
Vulkan ranges are disjoint, so this bookkeeping path is incompatible with the
native buffer carving. Source0eb9b36 changes ONLY initialization bookkeeping:
one full true root at a time, through one READONLY zero-work storage pipeline.
No public model write is authorized; Wgpu still initializes untouched ranges
BEFORE actual native commands. Each native instruction keeps its original
pipeline, descriptor spans, access/barrier/copy plan and full original owners.
No alias/initialization validation is waived and no numerical consumer changed.

The actual generated readonly initialization WGSL parses/validates through the
same installed Naga and retains exactly one LOAD-only storage global; log
strata-native-root-init-wgsl-20261015.log (small standalone Rust verifier source
native-root-init-validate-20261015.rs in evidence directory). Frozen0eb9b36
release passes, SHA25610d969e803d535e3e5dc6f308045e8e62dda779361d307461a931a8e4811a32e,
strata-native-root-init-build-20261015. Protected original whole-model completion
and final strict core/off Clippy running at writing. Do not infer resolution of
the GPU failure or complete capture/numerical parity from static WGSL alone.


## Prompt expert logical fault view and overnight stop (2026-10-15)

This supersedes the earlier pending root-init checks above. Exact0eb9b36
core/off strict Clippy pass. Both its protected run and the subsequent6d7f037
run get past the previous upload/submission validation failure and through
initial prompt GPU preparation/router work. Neither completes inference.
6d7f037 adds fixed-public source-location tags for explicit prompt layout
rejections, returning the same original Error::InvalidBufferLength. It emits
no values/content and does not relax guards. Frozen release:
strata-native-prompt-layout-trace-build-20261015,
SHA256473aff7b8d1d4f8430a358521508c8e4c9991db83d0f89091b219e79c4fe0065.
Protected evidence: strata-native-prompt-layout-trace-check-20261015 and
strata-native-prompt-layout-trace-root-run-20261015.log.

The first guard is strata_prompt_experts.rs:235: fault.size_in_used()!=4.
Its actual strata_global_moe::enqueue_data prompt caller passed full(fault),
the true 32 KiB root, instead of the four-byte logical view. Source da7a74e
changes that argument to u::handle(arena,fault). Full root ownership and cleanup
stay in the arena; the consumer guard is unchanged.30f5220 removes the stale
Error import left by the earlier tracing conversion. No GPU run has tested this
latest correction: user reclaimed the cards. Next physical gate, ONLY after
the user returns them, is the same protected original completion fixture, not
a speed comparison. Do not equate successful GPU router work with full-model
numerical or complete native capture acceptance.

Final read-only state strata-native-gpu-release-state-20261015.json at
05:38:22 UTC confirms Intel50,688,000 bytes used, zero NVIDIA compute owners
(NVIDIA desktop allocation591 MiB), zero all five classified GPU fault counts,
normal/unpaused Kuno admission, and Kuno/Ollama/Vashti HTTP200. All protected
workers exited. This is physical process recovery, NOT accepted owning erasure.
No production binary/config/model/driver change, GPU reset or reboot. Both GPUs
belong to the user until explicitly returned; all remaining work is CPU-only.
Goal remains paused, at the user's request; stop at this documented checkpoint
for the night. Whole rewrite remains incomplete. Stable whole target/MTP branch
capture/warm-up, mapped observation and CPU doorbell/plan service, native mapped
parameter publication, event/commit overlap, arbitrary request input and final
numerical/state/recovery audit remain required before throughput comparisons.

Exact source30f5220 final CPU checks pass: locked release strict Clippy for
strata-intel-core and feature-off strata-rust-port, both with -D warnings;
cargo fmt --check and git diff --check pass. Logs:
strata-native-logical-fault-{core,off}-clippy-20261015.log. The first exact
core attempt caught an unused Error import left by tracing;30f5220 fixes it,
and both final checks pass. The existing duplicate-bin-path Cargo warning is
unchanged. No full core suite, new numerical acceptance or GPU test is implied.


## Resumed selected Intel execution path (2026-10-15)

The user returned the GPUs and authorized continued work; the earlier overnight
stop is superseded. Source18ce0ae freezes as
strata-native-logical-fault-build-20261015, SHA256
2a65bc5bd71f62c7bb6c0d2be0074c54db89906421dff80a5a6ed2112ddc6030.
The protected check passes the prompt expert fault-view rejection and then stops
at InvalidFormat("matrix rank"). Exact public original shard header inspection
confirms all48 `blk.N.ffn_gate_inp_shexp.weight` are rank-one[2560], BF16.
Pinned sycl/src/prefill/prefill.cpp treats them as a one-row BF16 GEMM. b17bbb0
adapts ONLY this named original native vector in prompt/BF16 projection, and
checks its expert0/first0/rows1 packed-source request before cache lookup. The
weight offset no longer indexes an absent second dimension. Generic Matrix
rank constraints are unchanged. Final build/tests/physical acceptance pending.

The same source review uncovered the selected-mode mismatch described at the
start of this guide. Evidence: pinned sycl/serve/strata-sycl.sh defaults;
sycl/src/kernels/cuda/verify_kernels.dp.cpp resident_plan_kernel uses either cache
slot address or the authentic HOST mirror address; sycl/src/core/verify.cpp's
no_host loop condition skips host service; the retained comparison guide's
Verified runtime containment explicitly records device-plan/no-host. The
in-progress native core correction removes the per-layer CPU plan/readback and
stage fetch from decode, not the original HOST complement or real selected
experts. In parallel, request-local stable branch parameter/recording ownership
is being connected through the existing runtime ledger and full eraser. Do not
claim implementation, whole rewrite or speed acceptance until those commits and
checks are recorded.

After the failed18ce0ae fixture, Intel again returns to50,688,000 bytes; no NVIDIA
compute owners, all classified faults unchanged at zero, normal admission and
Kuno/Ollama/Vashti HTTP200. Failed owning cleanup remains unaccepted. No production
service, model, driver, installation or API change. Only public compiled fixtures
are used; private content never enters these receipts.

## Selected serving-mode audit and resumed completion checks (2026-10-15)

The user returned both GPUs and requested continued implementation. No throughput
comparison or speed-based selection is authorized before the whole-port gate.
Source audit now uses the retained tested serving configuration, not generic
program defaults or optional CUDA paths. Evidence remains under
`strata-comparison-evidence-20261009` in the external development directory.

- `sycl/serve/strata-sycl.sh:22–25` defaults DEVICE_PLAN/NO_HOST to1. The
  resident-plan shader resolves actual cache or HOST addresses into the primary
  list and writes zero secondary groups (`verify_kernels.dp.cpp:2097–2207`).
  `verify.cpp:1936–1970` skips host servicing in this mode. Source36d13d3 ports
  that connected behavior, keeping original source leases and real role stamps.
  Existing staging and CPU scratch budget reservations remain, unused by decode.
- The only expert-usage increment is inside the host callback
  (`expert_source.cpp:2578–2580`), skipped by NO_HOST. The scheduled every4-window
  adaptive pass has no eligible usage>=2 candidates. Dynamic promotion is not a
  demonstrated contributor to the measured serving result.
- Context32768 selects no prompt borrowing unless explicitly overridden
  (`generate.cpp:2144–2151`). Explicit prefill4096 bypasses the auto2048 fallback.
  The eight raw/two dequant prompt slots apply at this nonresident fraction.
- Both target and MTP use requested INT8 KV (`mtp.cpp:388–414`). Selected CJK
  draft vocab has106299 IDs, gathered original head rows (`mtp.cpp:749–778`).
  These remain real pending differences; the misleading English-named helper
  alone does not connect the selected CJK path.
- The serve branch exits before `ver.warm`/`mtp.warm` (`generate.cpp:11506,12150`).
  Serving lazily retains complete branch captures; that reuse is required. Single
  GPU async commit is enabled at3115 and joined at explicit state boundaries.
- All777 required DASLab original float tensor headers match `iq_pack.py::FORM`.
  Generic dense conversion is unnecessary here. Target embedding uses HOST in
  source2638–2647; current shared-weight GPU embedding placement is a remaining
  budget/residency substitution.

### Completed prompt-prefix receipt

Source`b17bbb0` admits only exact original BF16 `[2560]` shared-expert gate
vectors as one-row projections. All48 source headers have this shape; generic
matrix-rank validation stays strict. The frozen artifact in
`strata-native-shared-gate-build-20261015` has SHA256
`4a8a29d09a61b012123c40120ac3cac7c296984534c66e9abbf5f86f08709cae`.
Release build and18 focused native_shared tests pass.

Protected `strata-native-shared-gate-check-20261015` completed the original
batched prompt prefix including MTP teaching and reached last_prompt_input.
The final count1 prompt token then returned `InvalidBufferLength`; generation
and numerical parity are **not accepted**. Owning target/draft erase receipts
were not accepted. Process recovery was verified separately: admission restored,
HTTP200, Intel50,688,000 bytes, no Nvidia compute, fault counters unchanged.

Source`36d13d3` no-host integration builds. Frozen artifact in
`strata-native-no-host-build-20261015` has SHA256
`5d72b4d20322cb55390fa062b44fdc3c87f42927c203704f7143f5070aa2f240`.
Its whole-fixture completion check is pending; no speed result is claimed.

### CPU-only continuation after the GPU stop request

The user explicitly withdrew further GPU tests due usage and then chose to
leave the goal active for CPU-only work. Last protected36d13d3 run ended with
the same final-head `InvalidBufferLength`, no whole-generation receipt, and
separately verified physical recovery at06:07:22Z. No further GPU run follows.

Source`8c239bc` fixes that exact static boundary: the batched head bridge first
authenticates the full fault owner, then derives its four-byte logical view for
`strata_dense::prepare`. Erasure still covers the full allocation.

Source`19aa565` pins request-local original INFO pages by static branch shape,
nonce and submission segment. Record sizes/counts, pages and actual graph resource
signatures must match on repeat; named recordings survive the incidental LRU.
Real MTP round/step consumers are connected. Position-dependent attention grids
use admitted fixed context with original logical-visible guards. Source`7aa574f`
connects target AllRowsWithCarry verification captures by count and actual context,
excluding the final prompt token before request reuse. Source`cd8993d` stages PLE
OLD/EMB and source lookups before target capture; its armed prepared receipt is
consumed exactly once by layer1 math. Original source/state proofs stay required.
These changes do **not** yet eliminate rebuilding engine enqueue calls. True
direct replay still needs typed invocation parameters and all-family transaction
preparation without stale request positions. Cross-request retention is not yet
provided; original request owners are erased at request close.

CPU-only receipts in the external development directory:
- `strata-native-branch-params-build-20261015`:19aa565 release build, immutable
  SHA256`6278c613fbfb67c626a6e37518e74539b331cd4be75c2034b42c9626ca16cce4`.
- `strata-native-branch-clippy-20261015.log`: strict selected-core Clippy through
  7aa574f passes.
- `strata-native-branch-root-tests-20261015.log`:3 actual vendor branch-layout
  tests pass via root CPU-only harness (`683c97f`). Direct dependency package
  test invocation was rejected by Cargo because vendor crate is not a workspace
  member; no test executed in that attempted invocation.
- `strata-native-no-host-shaders-current-validator-20261015.log`:317 shader
  variants plus cooperative-matrix lowering pass using the pinned current
  validator in `spirv-tools-matrix-20261014/build/tools`. The earlier invocation
  without this PATH failed because system spirv-val lacks capability5116.

PLE source throughcd8993d is reviewed/formatted; its combined compile/test receipt
is pending. Concurrent INT8/CJK/typed-invocation work remains isolated, not current
accepted behavior. No installed service/model/API/UI change or speed claim.

### Final user-requested stop, selected core source4bf3ed9 (2026-10-15)

After usage reset, the user explicitly requested finishing the current work and
stopping. No further GPU work was started after the earlier CPU-only request.
All compilation/check subprocesses have finished; goal is paused at this stop.

Integrated `2a3d1ad` ports original INT8 target/MTP cache arithmetic together:
64-value groups, FP16-stored scale from amax/127, nearest-even/clamped signed
codes, and F32 code-times-stored-scale reads. Prompt/verify/index pooling, append,
guards, rewind, rejected-suffix handling and budget consumers are updated. Target
cache allocation decreases96MiB across12layers; MTP logical data shrinks but its
original allocator-rounded private owners remain4MiB. Token-contiguous target
transaction rows and per-head interleaved MTP scales remain explicit layout
substitutions versus upstream separated planes. Numerical equivalence is pending.

Integrated `4bf3ed9` keys target captures by the actual original GDN state-buffer
roles. Both recurrent and convolution pairs must align, across all GDN stages.
This uses two static physical binding arrangements, not token positions/contents.
It preserves Kuno's existing commit/restore swaps; upstream fixed-address commit
is still a documented scheduling substitution. No stale binding replay allowed.

CPU receipts: `strata-native-int8-check-20261015.log` passes release check through
2a3d1ad; `strata-native-int8-final-clippy-20261015.log` passes locked release strict
core Clippy at4bf3ed9, including the PLE staging/capture integration. These are
not shader/numerical/GPU validation for the newly changed INT8 code. The317-shader
receipt above predates INT8 and must not be counted as INT8 acceptance. Existing
updated layout/unit tests still need execution for this final source. No
feature-off final regression was rerun in this stop block.

Next: inspect the isolated CJK source checkpoint before integrating; complete
typed invocation parameters and GDN/QSA transaction preparation, then true direct
graph launch without recreating enqueue calls. Current branch reuse still walks
the engine to refresh parameters and validate exact signatures. The whole-port
source audit and original full-model numerical/owning-erasure acceptance are
incomplete. No speed comparisons until the entire completion gate passes.

Last physical evidence remains06:07:22Z, external
`strata-native-no-host-check-20261015/host-after.json`: normal admission, all three
HTTP200, Intel50,688,000 bytes, no NVIDIA compute and fault counters unchanged0.
Installed binaries/configuration/model files untouched. GPU tests require renewed
authorization; cached sudo is not permission. Required Vashti changes:none.

Isolated CJK source is saved cleanly as`cd703bc` on`codex/strata-native-cjk`,
worktree`/home/jmicko/.codex/worktrees/strata-resident-first/vashti`. Detailed
handoff: `docs/kuno-strata-cjk-checkpoint.md` in that worktree. It connects
original-order106299-row Q6 head, public ownership/budget and mapped coupled
sampling; no compile/tests/GPU acceptance. It adds224800768 public device bytes
and no private roots. **Not cherry-picked into main engine yet.** Next resume
review/cherry-pick onlycd703bc, preserve the distinct INT8 Append changes, then
compile and validate all compact-head/sampler variants. The two other peer
worktrees are clean and their completed source is already integrated.

## CJK and INT8 combined CPU integration (2026-10-16)

The active goal resumed; previous user restriction still limits work to CPU.
No GPU/model/service/sudo commands ran. The last physical recovery snapshot is
historical and was not refreshed; no new live runtime claim is made.

Original isolated CJK commitcd703bc is now integrated as`0bfbea8`, preserving
the distinct INT8 Append/capture changes. It adds the exact original-order
106299-row Q6 draft head, immutable IDs/inverse maps, compact argmax/confidence
and coupled sampler, while actual target verification remains full-vocabulary.
Head/IDs/inverse add224800768 public device bytes; no private roots are added.
Public loading gathers original Q6 GGUF rows in Rust rather than GPU gather, an
explicit startup substitution. Data remains in the original shared model files;
no converted model or request-content artifact is persisted.

Source`83035de` retains the checked full-vocabulary scratch calculator and adds
the actual compact full-page/tail Q6 specializations to offline shader coverage.
The tail7995rows is not a multiple of4, so its shader must preserve row bounds.
A later style-only map guard collapse resolves strict Clippy; checks below
distinguish the artifact source from that later source edit.

Receipts under `/media/jmicko/faststore/kuno-development`:
- `strata-native-cjk-integrated-check-20261016.log`: initial selected-core
  release check passed; unused scratch helper warning was corrected by83035de.
- `strata-native-cjk-test-build-20261016.log` and
  `strata-native-cjk-cpu-tests-20261016.log`:25 focused CPU cases pass. Each
  filter ran a nonempty test set:2fixture/plan,11sampler/request,2confidence,
  2targetINT8layout/math,4MTPfront,3branchlayout,1PLEphase. Full core regression
  and unrelated historical failures remain unaccepted.
- `strata-native-cjk-integrated-build-20261016`: locked release artifact at
  83035de, SHA256`455d5631e48a6ccd4d294d1dd1517a237a914eb38516292aaa3f77fec87b0fff`.
- `strata-native-cjk-int8-shaders-20261016.log`: all326 generated shader
  variants pass Naga/SPIR-V checks, physical expert-pointer checks and matrix
  lowering, using the pinned current spirv-val. This includes the new INT8 and
  CJK variants together. It initializes no device and performs no model run.

Remaining: typed invocation parameter staging, GDN/QSA transaction preparation
separate from enqueue, and retained graph launch without reconstructing kernel
calls. Commit/event overlap, source placement/configuration differences and the
whole connected audit remain open. New native startup, original full-model
output/state/owning-erasure checks require renewed GPU permission. No speed
comparison or complete-rewrite claim. Required Vashti changes:none.

Final strict checks at source`ed6373d` pass for both selected-core and
feature-off configurations: `strata-native-cjk-core-final-clippy-20261016.log`
and `strata-native-cjk-off-final-clippy-20261016.log`, locked release with
-Dwarnings. The first two Clippy attempts caught only the nested map-guard
style diagnostic; ed6373d resolves it without changing validation semantics.
Formatting of changed Rust files and git whitespace checks pass. Main checkout
is committed except the preserved inherited Python __pycache__ directory.
All compile/validation subprocesses have finished; goal remains active for CPU
source work. No hardware claim, installation or consumer API change.

## Attention invocation transactions before capture (2026-10-16)

Source`f965f63` connects preparation of every actual GDN/QSA transaction before
target graph capture. This follows the selected Intel control separation in
`verify.cpp::capture/run`: retain the window graph, stage the current invocation
inputs/metadata, then execute it and separately commit verified state. It is a
prerequisite of direct graph invocation, not the finished direct-launch path.

Each family now has an opaque PreparedGdnBatch/PreparedQsaBatch containing the
original armed receipt. Preparation authenticates graph/stage/public identities,
full private/public owners, ranges, shared-workspace/poison identity, context and
ordered per-stage generation; it starts the original lifecycle and stage anchor.
It queues no math, transfer or status kernels. The real core caller prepares all
48 attention stages before capture, using the original GR MIXED scratch identity.
The GR getter conveys only that identity at an idle workspace boundary, not a
produced value or completed-read authority.

Each actual layer's math enqueue consumes exactly its prepared transaction,
rechecks the original anchor/generation/state owners and matching input/output/
spans/public/fault identities, and returns the existing pending receipt. Kernel
math, arguments, original replay-cache chunk ID, status folds and final output/
fault/acceptance checks are unchanged. No prepared transaction can be passed to
commit: its inner receipt is private to the owning family. Abandonment retains
the existing sticky poison and full metadata zeroization. All preparation slots
must be consumed before the target branch seals. Legacy enqueue entry points
compose the same two phases, preserving their ordinary math schedule.

The fixed additional two48-entry option arrays are explicitly included in host
metadata admission; no GPU state, scratch or model roots are added. A focused
metadata-cap test covers the selected aggregate separately from older feature-off
hard-coded GPU totals. New prepared-ticket Drop tests exercise real armed receipt
abandonment in both families. Existing generation, alias, lifecycle, all-stage
preflight and commit tests remain the acceptance contract.

Initial CPU check caught a moved-local `chunk` reference in GDN replay retention;
it now comes from the authenticated original stamp, preserving its generation.
Strict selected-core release Clippy passes. Final CPU test build/results and
feature-off Clippy are pending at this receipt's creation. No GPU, service, sudo,
model mutation, image/private-content test or throughput trial ran.

Remaining direct-replay dependencies: typed invocation position/epoch staging,
retained native graph launch, and the original GR/shared/expert end-of-window
bookkeeping without rebuilding math. Prepared GDN/QSA/PLE tickets cannot replace
actual graph submission/completion, and an old graph must not run with stale INFO
positions. Commit/event overlap, placement/configuration substitutions and full
source/numerical/owning-erasure audits remain required. Required Vashti changes:none.

### Attention preparation CPU receipts

Exact sourcef965f63: locked release test build passes
(`strata-native-attention-invocation-final-test-build-20261016.log`). Focused
`strata-native-attention-invocation-cpu-tests-20261016.log` passes38cases:18GDN
batch,16QSA batch,1selected metadata-cap/inventory,3native branch-layout. Both
new prepared-ticket Drop cases execute. Existing generation/order, alias/epoch,
all-stage preflight-before-commit, cursor, queue/status, cancellation/poison,
owned-layout and shader-build cases pass within these named modules. Full native
core suite remains unaccepted outside this scope; no hardware execution implied.

Strict locked release core and feature-off Clippy pass atf965f63:
`strata-native-attention-invocation-{core,off}-clippy-20261016.log`. Formatting
and whitespace checks pass. Kernel shader sources/math were not changed; the
previous326-shader receipt is unchanged-source evidence, not a new live shader
execution. All checks are CPU-only and all processes have completed.

Next direct-invocation implementation must also preserve GDN replay-cache
metadata (start/count/original chunk), actual GR/shared consumer bookkeeping and
MoE stage epochs, not only CPU family anchors. Captured uniform pages need typed
position/epoch publications; patching bytes by comparing an old position is not
a valid parameter contract. Retained graph submission must own its original
resources and real queue completion. Only the existing whole-model proofs may
commit these new invocation receipts. No speed test is authorized by this source
progress; continue the full completion gate.

## Typed target invocation parameters (2026-10-16)

Source`e91677a` connects typed position/route-epoch controls to the actual target
branch, preserving its static key and original request nonce. The native QSA INFO
launcher declares position word1; global route/reduce declare epoch word2. Other
INFO words must remain exactly the captured constants. Dynamic fields are never
discovered by comparing bytes to an old position or epoch. The target coordinator
supplies its real m.position and the active global-MoE epoch after preparation.

The original QSA positions_batch scalar depended on the current position inside
generic launch metadata. Native `Kernel::Positions` now performs exactly that
bounded contiguous position/TOKEN-status stamp using the same typed INFO record.
Its TOKEN and maximum-position constants derive from qsa_gpu's original geometry.
It conveys no completion/commit credit; subsequent append/fault proofs still gate
state. Other GDN/GR/PLE scalar offsets are not falsely tagged as positions.

Branch parameter records retain explicit field schemas, immutable template bytes
with declared dynamic words zeroed, and original uniform owner/offset/length. On
repeat, stale supplied controls, changed constants/schema/owner/offset or shape
fail before submission. Actual mapped upload bytes are rendered from the template
and fresh typed controls through a write-only callback; no readable mapping or
unsafe cast bypasses WGPU's write-only contract. The first compile caught that
write-only view distinction and the callback resolves it. No new GPU root exists.
Templates wipe their full CPU allocation capacity at request cleanup; active
controls are cleared at branch close/retirement and are never Debug/Hash/Serialize
or cache keys. Template payload is bounded to2MiB and included in the existing
shared4MiB metadata admission, along with record/schema capacities. Uniform roots
remain pinned in the original transient ledger and follow existing queue joins,
recording-reference release and full erasure. Template metadata is not an erasure
or device-completion receipt.

This is a connected target control-rendering path, **not complete direct graph
launch**. Current target calls still traverse/enqueue kernels and validate their
exact recording signature. MTP branches deliberately retain their existing
untagged invocation mode; typed target controls reject an MTP shape. Do not claim
MTP parameter-program completion or enable no-enqueue execution from this block.

CPU-only receipts in `/media/jmicko/faststore/kuno-development`:
- `strata-native-typed-invocation-check-final-20261016.log`: release check passes.
- `strata-native-typed-invocation-test-build-20261016.log` and
  `strata-native-typed-invocation-cpu-tests-20261016.log`:28cases pass:6branch
  layout/typed renderer,16QSA batch lifecycle,2QSA layout/math,4MTP front. The new
  tests check fresh output with identical recording key/original binding, zeroed
  dynamic template words, stale/static/schema/binding rejection, context/epoch/
  field/duplicate/budget bounds, and absence of retained payload after rejection.
- `strata-native-typed-invocation-build-20261016`: locked release artifact at
  e91677a, SHA256`9896b0212cf332422fa23e2c19e2a3a3915fa5fbebf0c6191a382b5bd6381db1`.
- `strata-native-typed-invocation-shaders-20261016.log`:327generated variants
  plus native matrix lowering and physical expert-pointer checks pass with the
  pinned current spirv-val, including the new position-stamping entry.
- `strata-native-typed-invocation-{core,off}-clippy-20261016.log`: strict locked
  release checks pass; changed-file formatting and whitespace checks pass.

No GPU/model/service/sudo work, private-content test, or speed comparison ran.
Full native numerical/startup/state/owning-erasure acceptance remains pending.
Next: direct retained target submission with GR/shared/expert/replay bookkeeping
and original armed receipts, then the MTP round/step control program and direct
submission. Retain commit/event, placement/configuration, arbitrary-input and
whole-source audit requirements. Goal active, CPU-only. Required Vashti changes:none.

## Direct retained target graph submission (2026-10-16)

Engine `d40442c` connects a real no-enqueue repeat target path. Pinned Intel
`verify.cpp::capture` records the window once and `Verifier::run` stages fresh
controls before launch. The Rust backend now likewise submits the actual retained
Vulkan command inventory, resource grants and original compute/copy queue plan.
First execution still constructs the original graph; normal repeats bypass the
model's math traversal, kernel creation, descriptors, nodes and signature rebuild.

`Parameters::replay_key/begin_replay` require the exact request nonce, target shape,
typed sealed single segment, complete original pages and bound templates. Fresh
expressions preflight before mapping. Full original INFO pages get zero padding,
static template bytes and fresh position/route epoch/verification-usage values via
write-only callbacks. No per-kernel INFO allocation or per-record kernel enqueue
occurs on the direct path. Upload sources remain in the original mapped/unmap,
charge and full-erasure ledger; no new GPU root is added. Original template bounds
(2MiB) and shared metadata admission (4MiB) remain enforced.

`WgpuStream::replay_native_strata_branch` drains prior gathers/clears/PLE staging,
checks the real recording, stages/uploads parameters, waits for actual upload
completion, submits `Graph::replay_recording`, joins the native fence, and closes
the branch. The recording retains its original resources and compute count; partial
submission failures preserve the existing conservative in-flight ownership.
True only means this actual execution completed. False is only a pre-replay miss;
errors are terminal to the request and never fall back to fabricated execution.

The target coordinator checks original source publication, prepares all original
attention and PLE transactions before replay, and consumes them only after the real
submission. GDN input-cache copies are part of the recording; their CPU start/count/
chunk stamps are updated to exactly the current receipt. GDN original old/next role
must match. QSA original policy/anchor/caches remain checked. GR and shared workspaces
start/end idle with all read/combine/prepare/finish consumers present in the original
recording. Global MoE advances only queue bookkeeping, then still reads and verifies
all48 actual count/epoch/reduction/layer stamps. The unchanged output and two fault
proofs, family preflight, accepted-prefix commits and final erasure remain mandatory.
The new private joined-submission token grants none of those permissions itself.

Source audit caught two previously untagged positions: native GDN INFO word1 and
QSA radix-selection word1. GDN AB closing status has constant zero instead, so it is
explicitly not tagged. Route INFO word3 is a separate `RouteUsage` control: target
verification toggles that flag, and replay must not reuse the captured old value.
The reduction packet's unrelated constant zero is not tagged. Active controls and
static CPU templates retain their existing wiping and non-logging behavior.

Scope: nonzero-position AllRowsWithCarry target windows on active request reuse,
with a complete typed single-segment recording. Position0 initialization is left
uncaptured. Lab numeric observations/cancel hooks/grouped-status experiments keep
the existing enqueue path. Unsupported/multiple-segment recordings cannot be
flattened: there may be host observations between segments. MTP remains untyped;
its round/step input staging, observation, position and sampler/history bookkeeping
must be ported together next. Commit/event overlap, source-placement/configuration
substitutions, arbitrary input and the final whole-source audit remain open.

CPU receipts under `/media/jmicko/faststore/kuno-development`:
- `strata-direct-target-replay-final-test-build-20261016.log` and
  `strata-direct-target-replay-final-cpu-tests-20261016.log`:52 tests pass,
  including11 actual vendor program tests. The five added cases cover sealed/typed/
  nonce/segment admission, original multi-page rendering and zero padding, invalid
  owners/extents before writes, complete expression preflight and usage-flag refresh
  without changing keys or growing metadata. Existing GDN/QSA lifecycle, GDN math,
  MTP front and admitted metadata cases pass. The final source-publication guard is
  typechecked by selected/off Clippy and included in the immutable release build;
  these CPU tests do not exercise device execution.
- `strata-direct-target-replay-final-{core,off}-clippy-20261016.log`: strict locked
  selected-core and feature-off release checks pass. Changed-file formatting and
  whitespace checks pass. The first test script caught a nonexistent metadata-test
  filter and was corrected; zero-test runs were not counted as acceptance.

- `strata-direct-target-replay-build-20261016.out` and its artifact directory:
  immutable locked release at `d40442cef7a0536f32fa4f2eb68531165daaa705`,
  SHA256 `68fc3570ef389f15c2be4bf0619e2f18d8fd36b6486d4032c80493559dad52c9`.
- `strata-direct-target-replay-shaders-20261016.log`:327 generated variants plus
  cooperative-matrix lowering and physical expert-pointer checks pass using the
  pinned current spirv-val. CLI returns before vendor/adapter/device setup.

No GPU model,
private-content test, numerical/model acceptance, speed trial, install, service or
sudo action occurred. Full rewrite is still incomplete. Required Vashti changes:none.

## MTP input staging and actual observation boundaries (2026-10-16)

Source `862f5fa` follows pinned Intel `mtp.cpp::capture_round/capture_step` and
`MtpDrafter::draft`: prepare actual inputs/position controls before graph launch,
then inspect actual proposals at the confidence boundary. Source changes stay in
the Rust MTP owner/window and request sampler; target direct replay is preserved.

`prepare_native_front` performs the original25-root clear, TOKEN upload and actual
borrowed target-row or host-carry copy before the first captured front. Its opaque
`PreparedNativeFront` fixes model binding, all original root IDs, start/count,
owner cursor and source-copy mode. Only the same owner/cursor/root inventory can
consume it. Legacy scalar/teacher/repair callers compose preparation and enqueue,
retaining their exact mathematics and counts. Queued followup copies remain device
operations in the original chain. The owner-local pending flag stays set on receipt
abandonment/panic; it rejects a second preparation and publication. It leaves the
shared context usable for the original failure cleanup. The fixed receipt/atomic
metadata fits the existing2MiB validation/stack allowance; no new private GPU owner,
model mirror, persistence or erasure authority is added.

`Scope::begin_round` now stages all supported coupled draft controls once in its
original full32KiB CONTROL root. Each valid draft gets the exact original tuple
`(cell,j,round base)` in a disjoint aligned4KiB row. Out-of-context future rows and
all padding remain zero. `draft_pick` borrows that row instead of performing a host
write during recorded math. The unchanged GPU bridge checks actual device positions,
reads the original token/history, constructs params and publishes actual RNG draws,
CJK original IDs and probabilities. CPU phase/counter updates remain in the original
consumer. The complete owner retains its existing reservation and erase/audit path;
views never become independent owners. This is an explicit Vulkan staging
substitution for the original mapped step controls, using an already reserved GPU
root with all controls uploaded before capture.

Window packet sealing is queued after its real producers. Graph end/fence precedes
actual private packet readback, full padding/fault/source/ledger checks and the
confidence decision. Zero threshold retains its single final observation; positive
threshold retains each cell's actual observation before any followup. No readback
is silently recorded as replayable math. The original target residual grant remains
borrowed until the actual completed window observation, then its original caller
marks completion. Final forward publication also rejects an unconsumed preparation.

This does **not** implement MTP direct invocation. Existing MTP branches remain
untyped and traverse/enqueue the original math. Remaining typed controls need
explicit accepted-front/rest/follow-step phases, original guard boundaries and
actual request sampler/position bookkeeping before enabling the shortcut. Do not
reinterpret every INFO word1 as a position: native Select/Identity/closing packets
have other semantics, and packed-cache guard boundaries may equal4096. No timing
report or intermediate performance choice changes the selected source direction.

CPU receipts in `/media/jmicko/faststore/kuno-development`:
- `strata-mtp-staging-check-20261016.log`: selected-core release check passes.
- `strata-mtp-staging-test-build-20261016.log` and
  `strata-mtp-staging-cpu-tests-20261016.log`:26 tests pass:3 preparation lifecycle,
  4 confidence/source/packet,4 request sampler/staging/offline bridge,4 native front,
  11 actual vendor parameter-program cases. New cases cover exact owner/root/cursor
  checks, abandoned/panicking preparations, bounds before arming, all supported
  teacher/selected cursor shapes, complete staged-control rows/counters/disjointness,
  context-end inactive rows and full padding. Nonexistent initial window-test filter
  was caught and corrected; zero-test runs were not counted.
- `strata-mtp-staging-{core,off}-clippy-20261016.log`: strict locked selected-core
  and feature-off release checks pass. Changed-file formatting and whitespace pass.

- `strata-mtp-staging-build-20261016.out` and its artifact directory: immutable
  locked release at `862f5fa991a0f01ada832642b2d064d4d5ae8350`,
  SHA256 `65c91cad76737a4fa8097c2a2824eaa773aedb778593861dfe61691fff5b89e2`.
- `strata-mtp-staging-shaders-20261016.log`:327 generated variants plus native
  cooperative-matrix lowering and physical expert-pointer checks pass using the
  pinned current spirv-val. CLI returns before vendor/adapter/device setup.

All launched processes completed; all checks are CPU-only. No whole-model/numerical/GPU/erasure acceptance, private-content test,
throughput trial, driver reset, service change, install or sudo action occurred.
Existing full-core regression limitations remain. Goal active; required Vashti
changes:none.

## Typed MTP phases and original model namespace (2026-10-16)

Source `8262367` connects typed MTP controls to the actual `resident_mtp_window`
round/step consumers and native front/QSA/packed-cache launchers. Pinned Intel
`mtp.cpp::draft` stages front cells p+t, selected rest at p+a and subsequent cells
p+a+j; `capture_round` fronts accepted rows then runs the selected rest, and
`capture_step(j)` uses one current cell. Rust declares exactly those shape phases.

`NativeStrataInvocation::mtp` is separate from target controls. Its round range
covers rows+proposals-1 cells; independently observed steps cover one cell regardless
of their original step index. Target controls cannot authorize draft families or
vice versa. Target route epoch/usage sources reject MTP invocations. Actual origins
stay outside keys and retained templates; the retained active copy still wipes.

`NativeStrataPhase::Front/Draft{index}` resolves from immutable branch geometry.
The server executes preceding scheduled launch metadata with the old phase before
updating the next one; the phase declaration has no explicit GPU flush/join. The
launcher declares PhasePosition/PhaseBoundary fields, which `record_info` resolves
on a fixed8-field stack array to canonical Position/Boundary displacements before
record validation/capture. Repeats must match the same canonical schema/constants/
original owner bindings. Bulk replay needs only fresh invocation origin because
captured fields no longer contain unresolved phase expressions. Metadata still
uses its existing cap and templates zero dynamic words and wipe their full capacity.

Rounds explicitly switch front origin p to selected rest p+rows-1, then each
whole-chain follow-up p+rows-1+j before its clears/guards/front. Independently
observed step j starts at its current cell and must declare that same original j.
The actual front/positions/append/QSA/Identity records receive declared positions.
Identity reads start when filling its identity selection packet, so it is included;
Select's zero has other meaning and is not tagged. Rewind remains before capture.
Packed-cache guards separately declare current offset0 or closing offset1. Readable
cells remain0..4095; boundary4096 is valid and never becomes a readable token cell.

Core generic fused-sweep cache_position is provably unused: Keys/Values have
INTEGER roles and no other role is CACHE. Code verifies those original roles,
normalizes only that unused scalar's INFO to zero, and retains all original
full-owner/padding/finite/fault predicates. The separate packed-cache guard still
uses its actual typed boundary and original packed math. Feature-off cache roles
retain their original dynamic scalar. No numerical kernel or state proof is replaced.

The original C++ drafter owns its graph arrays per model instance. Rust uses a
shared target/draft stream, so actual MTP keys now include `self.binding`, its
existing immutable nonzero instance identity, in addition to shape/request nonce.
Two instances cannot reuse one recording even with identical geometry on the same
request stream. This is content-free model identity, never prompt/seed/token data;
first capture still authenticates and retains its full original resource inventory.

Scope: typed parameter publication is now connected, **not MTP direct invocation**.
The actor still traverses/enqueues actual math. Direct submission must preserve
original first-preparation consumption, cursor/dispatch/traffic state, request
sampler CPU counter/coordinates, actual per-cell confidence/fault observations and
final completed-forward publication. Original owning erasure and numerical/full-
model acceptance remain pending; no speed report changes implementation direction.

CPU receipts under `/media/jmicko/faststore/kuno-development`:
- `strata-mtp-typed-phase-final-test-build-20261016.log` and
  `strata-mtp-typed-phase-final-cpu-tests-20261016.log`:67 focused cases pass:
  15 actual vendor parameter tests,3 MTP preparation,4 confidence/source/packet,
  4 request sampler,4 native front,2 QSA layout/math,18 GDN lifecycle,16 QSA
  lifecycle and1 admitted metadata test. New parameter cases exercise all8x7 round
  shapes at several origins including context end, original steps1..6, canonical
  multi-phase bulk rendering without growth, position versus4096 boundary, invalid
  family/phase/extent/target-route sources before retention, and distinct original
  model/request namespaces. These are CPU program/lifecycle checks, not model output.
- `strata-mtp-typed-phase-final-{core,off}-clippy-20261016.log`: strict locked
  selected-core and feature-off release checks pass; changed-file fmt/whitespace pass.

- `strata-mtp-typed-phase-build-20261016.out` and its artifact directory: immutable
  locked release at `826236718ed42e3173f30a583d7e5e8ecc3c9965`,
  SHA256 `9ebe47ee4f7e121a667bba5e7ce6698f9558607e602d4dd6c0cb6b21c1393e51`.
- `strata-mtp-typed-phase-shaders-20261016.log`:327 generated variants plus native
  cooperative-matrix lowering and physical expert-pointer checks pass with the
  pinned current spirv-val; CLI returns before vendor/adapter/device creation.

All launched checks completed. No GPU, inference, numerical comparison, speed trial,
installation, service or sudo work ran. Full rewrite and
existing full-core regression triage remain incomplete. Required Vashti changes:none.


## Direct native MTP submission and original host handback (2026-10-16)

Source `d0670b01b8a964908fe1eba5f4d98b35ab872e39` connects the existing actual
single-submission replay backend to `resident_mtp_window::queued_window_with_minimum`.
Both whole zero-threshold chains and positive-threshold independently observed
steps attempt recorded submission before traversing their math. A miss retains
the original enqueue/capture path. An upload/submission/join error is terminal,
not fallback. Runtime still bulk-renders declared typed INFO fields into original
tracked resources and joins actual completion; no synthesized output/fault proof.

The actual actor's new private `strata_mtp_replay` module records immutable host
accounting only after a successful captured packet observation. Its original model
binding plus static round/step geometry and coupled-versus-greedy mode key separate
the original command inventories. Backend request nonce independently protects
recording availability. The fixed128-entry boxed table admits both sampler modes
for8x7 round shapes and6 follow-up steps; its conservative two-copy32KiB metadata
cap fits the existing2MiB host validation margin. No GPU root or private data cache
is added. Positions/tokens/seeds/carries/logits/history never enter this table.

Captured math may change only explicit-dispatch and actual follow-copy counters;
any actor host upload/read/observation/repair delta rejects the candidate. Replay
prepares checked counter additions from those actual deltas. Single cells preserve
the current per-token budget; full chains preserve the first token's actual budget,
intermediate resets and original final token counter. Context/copy/count/overflow
checks precede submission. Cache insertion occurs after real packet acceptance;
source readiness, original physical owners and cancellation are checked again.
The same prepared first-front receipt is consumed only after actual joined work.
Following positive-threshold copies remain at their existing boundaries.

Request sampler `DraftReplay` contains a wiping, request-local original-control
handback ticket. It checks actual phase3, original base/proposal coordinates,
counter, context and verify bounds. Only after the owning actor sees completed
native replay can it advance CPU proposal/counter state by the same original
per-cell rule. It verifies that original CPU state did not change first. The
original recorded GPU bridge/RNG/history/CJK mapping remains the sole producer of
sampled tokens. Greedy and coupled graphs have distinct keys, matching pinned
`sycl/src/core/mtp.cpp::capture_round/capture_step` graph arrays.

Every hit still calls actual `read_window_payload`: proposal/probability/fault,
padding, source and ledger checks are unchanged. Positive-threshold stopping uses
the actual current cell's probability, and the final original `commit_forward`
remains the only completed-forward publication. Borrowed target carry and output
leases keep their old lifetimes. `last_enqueue_ns=0` denotes no per-kernel actor
math enqueue for a replay; it is a diagnostic field, not device timing or the
controller's native draft-policy measurement.

Only fully sealed single-submission native recordings qualify. Source inspection
confirms the selected Intel runtimes in `main.rs` and `daslab_language_probe.rs`
set `tasks_max=32768`, with the native graph's same node bound; the legacy256-task
setting is not the selected mode. Unsupported/multi-segment branches safely enqueue
normally. This does not claim that every physical branch currently hits the cache;
that still needs authorized model execution after source integration.

CPU-only evidence under `/media/jmicko/faststore/kuno-development`:

- `strata-mtp-direct-replay-check-20261016.log` and
  `strata-mtp-direct-replay-test-build-20261016.log`: locked release check/test build.
- `strata-mtp-direct-replay-cpu-20261016.log`:73 focused cases pass:15 actual vendor
  parameter-program,3 MTP preparation,4 confidence/source/packet,4 new capture-derived
  replay accounting,6 request sampler,4 native front,2 QSA layout/math,18 GDN
  lifecycle,16 QSA lifecycle and1 admitted metadata case. New accounting cases
  cover single/whole budgets, original reset semantics, overflow, exclusion of
  host observations/repairs and shape/model/mode isolation. New sampler cases
  compare actual production handback helpers with original per-cell progression,
  and reject stale controls before mutation. No mock numerical/GPU output.
- `strata-mtp-direct-replay-{core,off}-clippy-20261016.log`: strict selected/off
  release Clippy pass. Changed Rust formatting and `git diff --check` pass.
- `strata-mtp-direct-replay-build-20261016.out`: immutable release at source above,
  SHA256 `a552e847a1c9cddec4ac7df4362c6b1ca7fcb76703b10eef38e36de4103920c3`.
- `strata-mtp-direct-replay-shaders-20261016.log`:327 shader variants plus native
  matrix and physical expert-pointer checks pass. The validator returns before
  vendor/adapter/device creation; this is GPU-free validation.

No GPU/inference/numerical/throughput/recovery/service/sudo operation, installation
or API change ran. Full-core regression triage, whole-model numerical/state/owning
wipe and physical acceptance remain open. Required Vashti changes:none.

### Next coupled source dependency: accepted-prefix commit and compute queues

Pinned Intel `generate.cpp:3115` enables asynchronous single-device commit.
`verify.cpp:2364–2483` enqueues the original accepted-prefix commit graph, signals
`commit_done_`, and defers its host join until an explicit session/final consumer.
Next target graph ordering protects its state; the independent MTP stream reads
only final window rows and its own K/V. Serving commit/draft/final-wait are at
`generate.cpp:11089`,11140 and11169 respectively. One-token self-commit and
all-accepted windows must be audited with the same path.

Current Rust `LatestTargetRound::verify` completes/commits the full proposed window;
`finish(n)` validates actual checkpoint owners, and rejection enters
`resident_target_round.rs::replay_prefix`: actual GDN accepted-prefix replay,
rejected QSA clearing, synchronous fault read, then family/cursor publication.
The controller drafts only after finish returns. This correctness-oriented schedule
is **not source-equivalent asynchronous commit** and remains an explicit port gap.

The native backend uses one compute queue plus its adopted transfer queue and one
in-flight fence. `strata_graph.rs::replay_recording/execute` call
`complete_submission` before the next work. A wait=false flag alone therefore still
serializes a subsequent draft. A coherent port must carry original pending state,
queue lane, actual completion and resource ownership together. Review actual
second compute-queue creation, target/draft classification, shared original carry
read dependencies, per-lane in-flight retention, accepted-prefix family publication,
and terminal/final joins before writing it. Do not repurpose the copy queue for
compute or forge completion to bypass original fault/source/erasure checks.

Other source substitutions in the checklist remain open; no speed benchmarks until
the complete connected source and correctness/owning-erasure gates pass. The goal
remains active for CPU-only work, not GPU or service permission.


## Dedicated native MTP compute queue and memory handoffs (2026-10-16)

Source `b6732c860cf6c5aaac050aecbd718af171bc07fb` connects an independent compute
queue to the actual native MTP branch capture/replay consumers. It does not remove
completion/observation checks or claim asynchronous commit overlap yet.

Historical physical source: `strata-matrix-capabilities-20261014.log` under
`/media/jmicko/faststore/kuno-development`, Intel BMG G31 at2374, family properties
3245–3282. Main family0 has one queue, family1 is compute/transfer and family2
transfer-only, one queue each. Those are saved Oct14 capabilities, not a live
query this pass. The production `backend/vulkan/strata_queues.rs::families` helper
examines actual properties on the future device creation: main must have compute,
another nonzero-count compute family is selected independently, and a distinct
transfer-only family is required. It prefers compute without graphics and does
not assume family IDs or request nonexistent queue1 of family0. Missing topology
rejects the selected experimental native device, rather than aliasing its queues.
Ordinary feature-off device requests do not request/adopt these extra queues.

The exact native `VkDeviceCreateInfo` requests queue0 of both selected families.
HAL records them once before allocation/exposure; getters borrow only actually
requested queues. Its buffer family list now carries an allocation-free valid
prefix of up to3 distinct families. All ordinary Wgpu buffers (including INFO
Uniforms and private storage), original host expert mirror, raw host weight pool
and prompt-shadow aliases use actual concurrent family sharing when adopted.
Generic nonnative imports keep their original exclusive behavior. No extra copy
of any model tensor or new private data root is allocated.

`WgpuStream::begin_native_strata_branch` flushes original work then selects the
actual branch queue. MTP round/step keys select the adopted compute-only queue;
target/unkeyed work selects the original queue. The graph switches command-pool
family, queue and timestamp capability together after real completion. Recording
identity/reuse now also requires its original family. Existing pipeline, full-buffer
initialization and retained-resource ownership remain shared, avoiding a second
initialization inventory that could invalidate another family's original roots.
Original copy nodes still use the actual independent transfer queue and its real
interval-dependent timeline edges. Direct replay selects the correct family,
submits the same retained work, then returns to the main queue after completion;
a missing recording safely returns to the original queue before normal capture.

Vulkan cross-queue visibility is explicit: `strata_handoff::Handoff` owns a timeline
semaphore and completion fence, reserves strictly increasing epochs before any
submission, signals the original source queue and waits at ALL_COMMANDS on the
actual destination. Branch switches carry original writes between compute queues;
fresh Wgpu primary-queue INFO uploads also hand off before the native draft graph.
A host fence wait alone is not used as a substitute for device memory visibility.
Partial failures retain synchronization handles until completion is proven; unknown
completion leaves them for terminal protected-process/device teardown. No driver
reset, process-external control or invented output/erasure proof is introduced.

Current scope/limitations:

- Only actual named MTP math branches switch to the independent queue. Unkeyed
  MTP prompt/preparation remains on the main queue and needs a source-scheduling audit.
- There is still one in-flight native graph fence; branch switches/actual packet
  observation still join it. This change does not claim simultaneous target commit
  and draft execution, nor any speed gain.
- Accepted-prefix repair still joins its original fault proof before returning.
  `TransientIo::prepare_batched_info` still uses Wgpu primary-queue copies/polling.
  Those are coupled remaining dependencies: a pending primary-queue target commit
  could serialize the next draft's INFO upload even with a second compute queue.
- Full physical numerical/state/recovery acceptance remains required. A build or
  a capability fixture is not proof that the current driver executes this path.

CPU-only receipts under `/media/jmicko/faststore/kuno-development`:

- `strata-draft-queue-test-build-20261016.log`: locked selected-core release test
  build passes. `strata-draft-queue-cpu-20261016.log`:40 focused cases pass:3 actual
  production queue-selection helper cases,1 actual epoch helper case,11 actual
  compute/copy interval planner cases,15 actual parameter-program cases,4 MTP
  replay accounting and6 request sampler cases. Selection tests cover saved B70
  shape, reordered/nonzero main families, flags/counts, missing or zero-count
  capabilities, and no silent queue alias. Epoch exhaustion never wraps/mutates.
  No queue/device is constructed by these tests.
- `strata-draft-queue-{core,off}-clippy-20261016.log`: strict locked selected/off
  release Clippy pass. Changed-source formatting and `git diff --check` pass.
- `strata-draft-queue-build-20261016.out`: immutable release source above,
  SHA256 `49ce70a6097ac50aa9b97541b7709fcc7d34a89be15b2652fa812e2fa91687be`.
- `strata-draft-queue-shaders-20261016.log`:327 generated shader variants plus
  native matrix and physical expert-pointer validation pass. This CLI returns
  before any vendor/adapter/device creation; the check is GPU-free.

### Next coupled block

Move actual full-owner INFO source-to-Uniform copies to native transfer scheduling,
with original mapping/ledger/schema/page ownership and exact semaphore dependencies.
`stage_native_replay` currently allocates/reuses a fresh Upload for each pinned
Uniform page: never retain a copy from one source handle while rendering new bytes
into another. Pair/pin the real tracked source with its page, or retain/validate
separate actual upload commands per invocation. Keep complete root/capacity wipe
and outstanding-use obligations. Direct replay must include the real current
copies, not just math that assumes unseen uploads completed.

Then connect original pending accepted-prefix work/fences, family permits/cursor
publication, target queue order, independent draft carry access and final/cleanup
joins as one coherent block. Remove full-proposed-window-before-prefix scheduling
only with its actual replacement. Do not simply remove fault reads, fake committed
cursors or treat a second queue as evidence of overlap. Other source substitutions
and the final whole-model/numerical/owning-erasure gate remain open.

No GPU/model/inference/sudo/service operation, installation, frontend/API change or
speed trial ran. The goal remains active for CPU-only work. Required Vashti changes:none.


## Native INFO transfer with actual current owners (2026-10-16)

Source `414e2c3c556e561187aa8570232d4e29391842c4` connects actual full-owner INFO
copies to the adopted native transfer queue in both cold execution and retained
math replay. No GPU or live speed execution ran.

`TransientIo::prepare_native_info` validates all original tracked source entries,
their Upload kind, mapped obligation and source/target full-size equality before
unmapping. It returns the actual current complete Upload/Uniform resources; all
bytes/pages remain under the original ledger and terminal wipe/audit obligations.
Typed replay still bulk-renders zero-padded original Uniform-page controls and
validates schema/nonce/model/branch identity. Upload allocations can change across
invocations, so the actual copy sources are not baked into a previous math recording.

New `strata_info_transfer::Uploads` owns a reusable transfer-family command pool/
command buffer, timeline semaphore and fence. Each invocation retains precisely
its current source/target handles and records full-size copies after HOST_WRITE to
TRANSFER_READ visibility. It checks original complete spans, COPY_SRC/COPY_DST
usage, nonzero/aligned bytes, distinct/null/duplicate/cross-alias roots and bounded
capacity. Timeline epochs reserve before submission and never wrap. Pending work
retains all resource/synchronization ownership on a partial submission error; an
unprovable terminal completion leaves those objects for protected-process/device
teardown, never free/reuse credit. Actual IO source allocations remain charged.

`strata_schedule::Commands::submit` places the actual INFO timeline wait at its
original compute header. That header's own timeline release propagates upload
readiness through the original interval-dependent compute/copy plan. Both cold
`Graph::execute` and direct `replay_recording` consume that same dependency and
join real final completion. The copy submission is retained until its own actual
fence; completed math is not fabricated. INFO-only drain has no math header, so
it explicitly establishes transfer-to-native and native-to-primary visibility for
subsequent reads/cleanup. Cleanup rejects outstanding copy work and includes it
in the original completion/retention boundaries.

This Vulkan substitution records one reusable actual-copy command buffer per
invocation, separate from retained math, because original Upload handles can vary.
Source bytes, full-owner movement, grouping and copy-before-consumer ordering are
preserved by the actual timeline edge. It adds a submission compared with putting
fixed mapped-input copies inside the original SYCL graph; final physical/source
acceptance must still review it. No optional default-off selector or other backend
is substituted for the connected Rust path.

`WgpuStream` now tracks actual primary encoder mutations: native root initialization,
ordinary/generic compute, readback, clear/copy, legacy private upload/uniform and
profiling writes. Native-only work no longer creates/submits/polls an empty primary
command buffer. Real primary work and pending write_buffer uploads still submit,
and real dependencies still join before native consumers. Feature-off flush retains
its original unconditional policy. Direct typed replay stages native INFO copies
without any new primary-queue submission. Actual final packet/fault/source/ledger
observation and completed-forward/family publication are unchanged.

Host admission changes: up to4096 copy pairs have a conservative additional2MiB
metadata reservation, including temporary span/hash sets and retained handles.
`NATIVE_QUEUE_HOST_BYTES` grows128MiB to130MiB; the existing original
`resident_model_prompt::HOST_BYTES` quote consumes that constant. No extra tensor,
private-content GPU root or untracked host staging allocation is introduced.

Source audit confirms Upload usage includes MAP_WRITE and initial creation is
mapped. Pinned Wgpu29.0.4 `device/resource.rs` maps those buffers directly as Active;
`resource.rs::unmap_inner` flushes noncoherent writes/unmaps the original buffer.
The unmappable Init-staging path that queues an implicit primary copy is not this
Upload path. First-use Uniform/storage initialization remains real primary work;
the new copy runs after that initialization, preserving original bytes and Wgpu
ownership tracking. Warm reused originals avoid that first-use initialization.

Evidence under `/media/jmicko/faststore/kuno-development`:

- `strata-native-info-test-build-20261016.log` and `strata-native-info-cpu-20261016.log`:
  release test build and43 focused CPU checks pass. These cover2 actual production
  copy-layout/capacity cases,12 original queue planner/dependency cases (including
  actual INFO-header/copy/compute dependency selection),15 actual parameter-program,
  4 MTP replay accounting,6 sampler,3 actual topology-selection and1 handoff epoch
  case. No GPU queue/device is constructed. GPU copy submission and INFO-only
  physical visibility are not established by these helper tests.
- `strata-native-info-final-{core,off}-clippy-20261016.log`: strict locked selected/
  feature-off release Clippy pass, including the final INFO-only drain. Formatting
  and `git diff --check` pass.
- `strata-native-info-build-20261016.out`: immutable locked release at source above,
  SHA256 `a7751e82073f1b6fc1bfca4d93110701b360786264c7cc173cdf31ef3270f11c`.
- `strata-native-info-shaders-20261016.log`:327 generated variants plus native
  matrix/physical expert-pointer validation pass. The CLI returns before any
  vendor/adapter/device creation; this is GPU-free validation.

The graph still owns one in-flight math fence and globally joins previous work.
Accepted-prefix repair/publication remains synchronous, and the current verifier
commits the full proposed window first. Next is the coupled pending-commit contract:
separate actual flights/fences/resources and family permits, accepted-prefix state
kernels, exact original carry access for independent drafting, dependent-target/
final/cleanup joins and failure retirement. Queue-switch handoffs must not signal
behind an unrelated pending target commit when only verified carry readiness is
required. Existing completed replay booleans remain completion authority and cannot
be reinterpreted as enqueue-only success. Unkeyed MTP prompt/preparation scheduling
and other listed source substitutions remain open.

Goal remains active for CPU-only work. No GPU/model/inference/numerical/throughput/
recovery/sudo/service operation, installation, public API or frontend change ran.
Full-core regression triage and whole-model numerical/state/owning-erasure/physical
acceptance remain incomplete. Required Vashti changes:none.


## Native lane fences and original recording flight ownership (2026-10-16)

Source `86f4c35d0b3e036a76571f9afc2cbb069d054ccd` replaces the native global
single-fence/last-recording completion assumption. Cold capture, signature cache
hits and direct typed replay all call the same actual `submit_last_recording`.

Each actual target/draft lane now owns a separately created fence. Each retained
recording has a nonzero monotonic identity, never a token/origin/seed. The fixed
2-slot `strata_flights::Book` stores that original identity, actual compute count,
replay/profile metadata and partial/submitted status. Before resetting any fence,
the owner validates the original recording family/count and reserves its free lane;
another live recording/lane cannot be overwritten or have its fence reset. The
slot is partial before queue submission and becomes submitted only after the actual
whole schedule submit returns successfully. Partial failure remains terminally
owned, never successful completion.

Original recordings retain every actual command/resource inventory. Only duplicate
enqueue nodes clear after successful submission. `pending()` includes in-flight
work as well as queued nodes, preserving original readback/cleanup drains. Cache
lookup/replay excludes live recordings, named replacement rejects a live owner,
and unnamed LRU eviction excludes live identities. A stable identity survives LRU
movement and other recording completion/removal; no completion/profile collection
uses the vector's current tail as authority.

`complete_lane_submission` looks up the actual recorded identity, validates its
original queue family and waits on that exact lane's fence. Only after real fence
success and applicable original profile observation does it clear matching host
bookkeeping and credit actual operation counters. Those counters are not inference
output/state authority; original fault/packet/ledger/family proof remains required.
Explicit cleanup drains all actual lanes before releasing recordings. An unknown
partial/fence failure retains command/resource/initialization inventories for
terminal protected-process/device teardown. Constructor failure paths and normal
cleanup own both actual fences. The bounded book's conservative4-copy representation
fits1KiB; recording IDs stay inside existing bounded metadata. No private tensor or
content root is added.

**Current limitation:** `complete_math_submission` still joins both lanes, and
public stream/graph switch/push/flush consumers still call the global drain. The
single INFO upload owner and queue-switch handoffs remain globally coordinated.
This source does not yet permit a target accepted-prefix commit to stay pending
while independent drafting proceeds. Do not remove global waits until its original
state/carry/INFO consumers have an actual dependency contract. Existing replay
booleans still mean completed work, never merely submitted work.

CPU-only evidence under `/media/jmicko/faststore/kuno-development`:

- `strata-lane-flights-test-build-20261016.log`, `strata-lane-flights-cpu-20261016.log`:
  release test build and46 focused cases pass:3 actual flight-book host checks,
  2 INFO layout/capacity,12 actual interval/upload-dependency planner,15 actual
  typed parameter,4 MTP replay accounting,6 sampler,3 topology and1 handoff epoch.
  New cases cover distinct simultaneous metadata slots, transfer-only counts,
  duplicate/foreign lane/identity/submit rejection without overwrite, partial work
  never clearing as joined, and exact-slot host bookkeeping. They create no GPU
  device/fence, mint no output receipt and do not prove physical parallel execution.
- `strata-lane-flights-final-{core,off}-clippy-20261016.log`: strict locked selected/
  off release Clippy pass. Formatting and `git diff --check` pass.
- `strata-lane-flights-build-20261016.out`: immutable locked release at source above,
  SHA256 `e3d4d1675d27308f8cca64c57242b58bb129c4bf99ced84f650accedcc4e9f68`.
- `strata-lane-flights-shaders-20261016.log`:327 generated variants plus native
  matrix/physical expert-pointer validation pass. This path exits before any
  vendor/adapter/device creation; it is CPU/offline validation.

Next coupled source block: original verified-window family permits and accepted-
prefix commit/publication, with only actual dependency joins and exact retained
carry access for drafting. `run_global` currently borrows GDN/QSA/PLE acceptance
across the final proof/commit. Preserve those policy/drop/poison obligations when
introducing deferred work; avoid self-referential stored borrows or casually
"disarming" originals. A scoped verified-prefix consumer while permits remain on
the stack, or explicit original-owner pause/resume permits, needs source review.
Upstream `capture_commit` uses dynamic n_keep; audit all-accept multirow replay and
count1 self-commit together with current full-window-before-repair scheduling.
Actual commit INFO must declare count/position fields explicitly; live uniform
targets cannot be overwritten. Normal target/session/final/cleanup consumers must
join their original pending state before using it.

No GPU/model/inference/numerical/speed/recovery/sudo/service operation, installation
or API/frontend change ran. Full rewrite, regression triage and physical/owning-
erasure acceptance remain incomplete. Goal stays active for CPU-only work.
Required Vashti changes:none.


## Native GPU PLE history and real prefix consumers (2026-10-16)

Sources `595393e9b173c879268d16b74344b02bd2048c16` and
`57147a8bd7c387affdd9535d027c39ed320861d3` connect the original GPU history
through native decode, prompt continuation and latest-window prefix consumers.
Pinned `verify.cpp` keeps PLE float history on device and advances only two ngram
token IDs on the host. The previous Rust decode instead uploaded CPU history,
read back OLD/NORMALIZED/NEXT and built another CPU float-history candidate;
prompt completion also rebuilt that mirror. Those default native operations
are removed together. An explicit grouped-read diagnostic still performs real
reads, and the feature-off CPU oracle retains its existing float checks.

`prepare_device_history_tokens` checks the original mutable token owner, exact
Qwen configuration, old pair, valid ordered inputs and cancellation. Its permit
only publishes two token IDs; it supplies no GPU/fault/completion authority.
The actual native caller still accepts original whole-model output/fault and
request-I/O proof, preflights all state families and preserves armed-drop/poison
obligations. Original GPU OLD survives preparation; GPU NEXT is a candidate.
Successful family completion swaps existing OLD/NEXT roles. Large prompt
completion uses the same token-only permit and existing GPU role swap; the
extra normalized-tail diagnostic copy runs only for actual legacy/diagnostic
consumers. No private float-state allocation is added.

Native latest-window checkpoints now retain token-owner/control metadata, not
a CPU convolution-history snapshot. Completed validation checks the exact original
root inventory with its OLD/NEXT swap. Actual rejection queues the existing GPU
History kernel over the original prewindow history plus accepted normalized rows;
zero-prefix restoration copies the real prewindow GPU history. The original joint
fault/audit fence still precedes queued token/cursor publication. Other families
remain part of the same preflight and commit boundary. Original generation,
sequence/binding/root, poison, cancellation, retirement and output-owner checks
remain; no checkpoint is disarmed merely because work was submitted.

Because native decode now also swaps PLE roles, real target capture keys encode
GDN bit0 and PLE bit1 independently (all4 inventories), while excluding content,
positions and expert selections. `native_capture_history_role` derives from the
actual original buffer identities. No new manual role state is introduced.
The immutable builder now freezes/checks `crates/kuno-text` too, since native
engine behavior depends on that crate.

**Still incomplete:** full proposed-window publication precedes accepted-prefix
repair, and repair/global queue switches still join synchronously. Pinned Strata
uses retained per-row PLE history snapshots and indexed commit; this port derives
the accepted history from original normalized rows and prewindow history instead.
That state/scheduling substitution, GDN role swapping versus source state commit,
actual pending accepted-prefix events, independent draft consumers and final joins
remain for the connected commit block/final audit. GPU residency of PLE history
alone does not establish source queue overlap or numerical parity. The legacy
`model.ple_conv.history()` is not authoritative after native history advancement;
never reupload it in a native continuation.

CPU-only evidence under `/media/jmicko/faststore/kuno-development`:

- `strata-ple-device-history-complete-final-test-build-20261016.log` and
  `strata-ple-device-history-complete-cpu-20261016.log`: locked selected release
  test build and61 focused native CPU cases pass (15 PLE preparation/owners,
  16 typed target/MTP parameters,3 PLE shaders,3 flight ownership,2 INFO layout,
  12 actual interval/queue planner,4 MTP replay,6 sampler). These create no device
  and supply no physical execution/completion evidence.
- `strata-ple-device-history-complete-core-tests-final-20261016.log`:9 original
  library transition cases pass, including the new mixed2048-row prompt/decode/
  short-prompt continuation against the real token-history implementation,
  every original decode prefix0..8, foreign owner/input/cancel rejection,
  dropped permits and unchanged legacy float history. The first new test build
  had two signed-token type mismatches; final log records their correction.
- `strata-ple-device-history-complete-{core,off}-clippy-20261016.log`: strict
  locked selected/off release Clippy pass. Changed-source formatting and
  `git diff --check` pass.

- `strata-ple-device-history-complete-build-20261016.out`: immutable locked
  selected release at `57147a8bd7c387affdd9535d027c39ed320861d3`, SHA256
  `7dab5174b04ae2826861dae8e29e6e41b8ff3adde277819924ace4cf2fe346d2`.
- `strata-ple-device-history-complete-shaders-20261016.log`:327 generated variants
  plus native cooperative-matrix/physical expert-pointer validation pass. The
  validator exits before vendor/adapter/device creation; this is offline CPU work.

No GPU/model/
inference/numerical/speed/recovery/sudo/service operation, installation, API,
frontend or release change occurred. Goal remains active for CPU-only work.
Required Vashti changes:none.


## Native accepted-prefix before state publication (2026-10-16)

Source `131d29c7feb02968f21c4694c54b4952c6d37db5` replaces the selected global
controller's full-window commit followed by checkpoint rewind with an actual
scoped native prefix consumer. This is connected in `strata_global_controller.rs`,
not a separate probe or default-off helper. Original prompt/teacher full-row
calls remain full-row calls; the diagnostic `LatestTargetRound` path still exists
but is no longer the selected native speculative controller.

Pinned `verify.cpp::capture_commit` retains original GDN inputs, commits n_keep
using the original recurrence/convolution, maintains QSA index state, and selects
PLE history. `commit` skips the graph only for count1 self-commit, and runs multirow
commit even when every proposal survives. `ResidentModel::verify_native_prefix`
now observes the actual target result while all original full-proposal family
transactions remain armed. Its scoped observer has provisional rows but cannot
borrow the model or executor. It derives acceptance from real target picks and
retains their original device carry. `select_verified_rows` releases the host row
lease before any prefix consumer; invalid extents/errors publish no cursor.

The real consumers are selected together:

- GDN checks each original pending receipt and its retained-input start/count/chunk,
  then reuses the actual state-only convolution/recurrence/fault consumers from
  prewindow OLD into uncommitted NEXT. The old diagnostic repair uses that same
  implementation with its opposite root roles. Multirow all-accept also replays;
  count1 already has its accepted candidate and needs no extra state graph.
- QSA checks the original pending owners and clears/folds only the rejected raw
  cache, positions and pooled suffix. All-accept queues no suffix clear. Existing
  packed raw-index storage supplies partial-block continuation; this remains an
  explicit storage substitution for upstream's index-tail snapshot/reappend.
- PLE checks the original pending owner and forms accepted NEXT from original OLD
  plus normalized accepted rows on GPU. Its original history permit then derives
  only the accepted two CPU ngram token IDs. No stale CPU float owner is used.

Original receipt metadata and policy accounting retain the full proposed count;
no receipt, generation or anchor is rewritten to imitate a smaller execution.
GDN/QSA acceptance permits select one bounded prefix; their real lifecycle commit
checks the original pending extent and publishes only the chosen next position.
PLE and model cursors publish that same n. Original real fault/I/O proof covers
prefix consumers before whole-family acceptance and state publication; receipt
abandonment, poison and terminal model/executor teardown still apply. Usage
observes the verified route window once and global adaptation remains after the
actual completed boundary. No new GPU state/carry owner is added.

Prefix INFO now coexists with verification before the request checkpoint. The
original runtime admission conservatively adds544,768 bytes: two audited copies
of2048-byte records for36 Conv/Replay/fault-fold triplets,12 QSA clear/fold pairs
and1 PLE History. The existing metadata/carry budgets retain original owners;
the selected controller no longer allocates latest-window checkpoint snapshots.

**Remaining source gap:** this block is synchronous. Prefix operations are actual
unkeyed enqueues, not yet Strata's retained dynamic n_keep commit graph. Original
fault/I/O proof and stream/queue drains still join; no target-commit/draft overlap
is established. Port the pending lane event and retained INFO ownership with
independent verified-row/draft-local consumers, next-target ordering and explicit
session/final/cleanup joins together. A scoped post-submit draft consumer can
retain original family permits on the stack; do not store self-referential borrows
or mint a completed cursor from an enqueue ticket. GDN/PLE role swapping and PLE
normalized-row reconstruction versus source indexed snapshots remain explicit
substitutions needing final source/numerical review. Other context, embedding,
arbitrary-input, regression, owning-erasure and physical gates remain open.

CPU-only evidence under `/media/jmicko/faststore/kuno-development`:

- `strata-direct-prefix-final-test-build-20261016.log`: locked selected release test
  build passes. `strata-direct-prefix-cpu-20261016.log` records98 nonempty passing
  cases:19 original GDN family,17 QSA family,1 scoped row-lease lifecycle,15 PLE,
  16 typed parameters,3 PLE shaders,3 lane flights,2 INFO,12 queue planner,4 MTP
  replay and6 sampler cases. The first controller filter was empty and is not
  counted; `strata-direct-prefix-controller-cpu-20261016.log` runs the corrected
  `daslab_language_probe::global_controller::tests::` filter with4 real passes.
  Total102, not the full regression suite. New actual lifecycle tests cover every
  1..8 proposed/accepted extent, direct continuation and invalid/stale rejection;
  generic row-lease tests prove host drop ordering, not GPU completion authority.
- `strata-direct-prefix-{core,off}-clippy-20261016.log`: strict locked selected/off
  release Clippy pass. Changed-source formatting and `git diff --check` pass.
- `strata-direct-prefix-build-20261016.out`: immutable selected release at source
  above, SHA256 `b6a9cb0290118c5238db942df05662964c98fb88d154e8287b254e85afce7b72`.

- `strata-direct-prefix-shaders-20261016.log`:327 generated variants plus native
  cooperative-matrix and physical expert-pointer validation pass for that same
  frozen artifact. This validator exits before vendor/adapter/device creation.
No GPU/model/inference/numerical/speed/recovery/sudo/service operation,
installation, API/frontend or release change occurred. Goal remains active for
CPU-only work. Required Vashti changes:none.


## Retained native commit graph and explicit accepted count (2026-10-16)

Source `4083258c9d505e26c9abdac35f5ba6516683ab2d` connects the selected native
prefix path to real lazy capture/direct replay, following the dynamic n_keep
principle in pinned `Verifier::capture_commit`/`commit`. `resident_native_commit.rs`
now owns the actual call from `run_global`. Every cold or warm entry preflights
all original armed GDN/QSA/PLE receipts, retained GDN input stamps/public origin,
physical inventories and exact original start/proposed extents. A cache hit alone
cannot authorize state publication. Proposed count1 returns with its original
one-row candidates and captures no commit graph.

Multirow keys contain original model-instance owner (the original PLE workspace
identity), proposed T, actual context and independent GDN/PLE root roles. The new
`NativeStrataInvocation::commit` owns position and accepted n; both are explicit
runtime INFO fields, never keys. `AcceptedCount` is accepted only for this commit
family. Existing target routing and MTP controls cannot be substituted for it.
Original active-control wiping includes the new field, and captured templates
zero the dynamic words as before. There is no request content in keys/templates.

All proposed-span command inventories remain fixed as n changes:

- GDN Conv and Replay declare accepted count and position. Conv dispatches fixed
  proposed T with its original runtime count guard; recurrence uses the accepted
  loop bound. A native64-lane finite/status fold uses the same original state,
  raw history, status and global fault owners. This replaces only the native
  commit's generated fold; old diagnostic replay retains its existing kernel.
- QSA clear and exact zero-fold are separate native kernels, consuming original
  packed raw cache/positions/pooled/fault roots with proposed T/position/n INFO.
  Both dispatch a fixed proposed span. All-accept is a true zero-length no-op
  inside those kernels, so a cold all-accept capture retains the later rejection
  consumers too. The original strict zero test and fault bit4 are preserved.
- Original PLE History declares accepted count and keeps its fixed history grid.
  It still reconstructs accepted history from normalized rows rather than using
  upstream per-row snapshots; that explicit backend substitution remains open.

A warm hit uploads actual fresh INFO and executes the original retained recording
without rebuilding per-layer nodes. A miss captures the same actual133-command
inventory (36 GDN triplets,12 QSA pairs,1 PLE). Existing full runtime INFO quote
544,768 bytes still covers it; no GPU state root is added. The real runtime API
currently joins the cold/hit submission. The original subsequent fault/I/O,
all-family acceptance, sticky poison and cursor publication remain mandatory;
this commit does not manufacture a pending event or erase completion obligations.

**Still incomplete:** actual async commit/draft overlap. `Stream::flush_inner`
still globally joins native work; branch begin/end/replay and `Graph` queue
selection/INFO upload/replay still join existing submissions. Selecting a new
queue currently signals a handoff behind main work, which would serialize a draft
behind pending commit. `resident_mtp_window.rs` also calls `start_window`, clears
its output and `prepare_window_first` before beginning its named MTP branch.
Draft TOKEN/carry staging therefore needs explicit independent-lane readiness too.
An event/INFO change that leaves those preparatory operations behind target
commit is incomplete. Preserve original per-lane recording/fence ownership and
actual verified carry sources; do not redo the completed stable flight book.

Next coupled contract: original verified rows ready before commit submission;
retained pending commit and INFO destinations; independent MTP preparation/math;
then the actual fault/state, next-target, session/final/cleanup joins. A scoped
post-submit draft consumer can keep original target transactions/permits on the
stack, avoiding self-referential stored borrows. It must receive only the original
verified carry, never a false completed-model cursor. Separate logical submission
from real GPU/fault/erasure acceptance. All other current checklist gaps remain.

CPU/offline receipts under `/media/jmicko/faststore/kuno-development`:

- `strata-prefix-graph-test-build-20261016.log`: locked selected release test build.
- `strata-prefix-graph-cpu-20261016.log`:104 nonempty focused native CPU cases pass
  (19 GDN,17 QSA,1 scoped lease,4 controller,15 PLE,18 actual parameter-program,
  3 PLE shader/oracle,3 flight,2 INFO,12 queue,4 MTP replay,6 sampler).
  New vendor program tests reuse the original page/template across every1..T
  prefix, all4 binding roles and positions at context end; they reject wrong
  families/extents/owners and show dynamic bytes absent from the template.
  One-model target+commit+draft key geometry has122 distinct shapes within the
  existing128-branch bound. These helpers grant no physical completion evidence.
- `strata-prefix-graph-{core,off}-clippy-20261016.log`: strict selected/off locked
  release Clippy pass. Changed-source formatting and `git diff --check` pass.
- `strata-prefix-graph-build-20261016.out`: immutable selected release at source
  above, SHA256 `c197a54d467147126da05da046f81571e51a5258ed73460db99e9fa884e75bc6`.
- `strata-prefix-graph-shaders-20261016.log`:330 generated variants plus native
  cooperative-matrix/physical expert-pointer validation pass, including the new
  GDN commit fault and two QSA commit variants. Validator exits before device
  creation. GPU numerical/visibility/execution acceptance remains open.

No GPU/model/inference/speed/recovery/sudo/service operation, installation,
API/frontend or release action occurred. Goal remains active for CPU-only work;
the whole rewrite is not complete. Required Vashti changes:none.


## Original pending commit submissions and exact joins (2026-10-16)

Source `91e1f9e130bb1a1359323f9a3b70c35435ac2f33` connects real pending cold and
retained target-commit submission to the selected engine. The new private runtime
client/server methods are `replay_native_strata_commit_pending`,
`end_native_strata_commit_pending` and `join_native_strata_submission`. Existing
`replay_native_strata_branch` still returns true only after its original join;
old branch-end behavior also remains joined. Pending APIs reject other branch
families and profiling (whose collector may consume/remove recordings).

The backend exports `NativeStrataSubmission` only after its actual queue submission
and original flight book report submitted. This is opaque graph/request-scope/
submission identity, not a numerical/model-ready/erasure receipt. Its public
metadata constructor is not authoritative: joining checks the original backend
export, graph, request reuse scope and original recording/queue lane. No Debug or
Serialize is added and private scope identities are never logged.

Every actual math submission receives a checked monotonic epoch separate from
its retained recording ID. Each lane permits one exported identity until ack;
an old token cannot join a later replay of the same recording. The book keeps
original joined metadata when an ordinary dependent drain completes the fence
before explicit ack. Other normal submissions cannot overwrite that completion.
Acknowledging consumes only that export and cannot clear a newer live flight.
Partial/non-submitted work cannot export or complete. Cleanup clears exports only
after all original math fences joined, and never resets the epoch. Four complete
copies of bounded flight/export metadata still fit the existing1KiB allowance;
no GPU tensor/state/carry root is added.

`Graph::join_exported` either waits on the actual matching lane's original fence
or consumes its already recorded real completion. `complete_lane_submission`
remains the sole production path that credits joined bookkeeping, after exact
record/family validation and real fence success. Cold/hit pending methods keep
original recordings, INFO and resources retained; they skip the joined path's
queue reset/uniform release. An occupied export slot rejects another pending
submission before new queue work. Ordinary failure still retires the original
model/executor, and explicit cleanup retains unknown/partial work.

The original flight now also stores the exact INFO timeline value its header
actually waits on. Only its real math-fence join credits that upload epoch.
`Uploads::needs_visibility` distinguishes a copy with no completed consumer from
a copy already consumed by explicitly joined math; a target fence cannot credit
a newer unrelated copy. This avoids adding a spurious INFO-only handoff at the
new join boundary. It does not itself provide cross-queue memory visibility;
original semaphores/global guards still apply.

The actual selected call is `submit_native_prefix_commit` followed by explicit
`join_native_strata_submission` in `run_global`, before original real fault/I/O
and all-family acceptance. Count1 returns no extra submission. **The engine still
joins immediately; no commit/draft overlap is claimed.** Keeping this join until
the coupled data/INFO/queue consumer exists preserves original state publication.
No ordinary API timeout, installed runtime, model or service configuration changed.

Next: establish verified-row readiness before commit, then admit an original
scoped drafter while target transactions/permits remain owned. `start_window`
clears/stages token/carry and checks private state before named MTP capture;
`prepare_window_first` adds more uploads/copies. Selecting the independent queue
only at branch begin is insufficient. Global graph push/transfer, stream flush,
queue selection, INFO-upload reuse and replay still drain prior math. INFO copies
are full-owner copies, so disjoint record offsets alone do not protect a uniform
page read by a live commit. Original data intervals and copy/compute header waits
must establish actual independence. Preserve target/session/final/cleanup joins
and all other open completion-gate items. Merely delaying the explicit join does
not complete this block.

CPU/offline receipts under `/media/jmicko/faststore/kuno-development`:

- `strata-pending-commit-final2-test-build-20261016.log`: locked selected release
  test build. `strata-pending-commit-cpu-20261016.log`:107 nonempty focused cases
  pass (19 GDN,17 QSA,1 scoped lease,4 controller,15 PLE,18 parameter program,
  3 PLE,5 flight/export,3 INFO,12 queue planner,4 MTP replay,6 sampler).
  New cases exercise stale/duplicate/wrong-scope exports, retained completion,
  replay epochs, both lanes, partial/cleanup/overflow and exact INFO attribution.
  These are actual host bookkeeping/layout tests, not Vulkan fence/visibility proof.
- `strata-pending-commit-{core,off}-clippy-20261016.log`: strict selected/off locked
  release Clippy pass. Changed-source formatting and `git diff --check` pass.
- `strata-pending-commit-build-20261016.out`: immutable selected release at source
  above, SHA256 `9d95206461715afad790e72e4bb42ca7a0ea8ed4e011472bdb7efafc6faa85f8`.
- `strata-pending-commit-shaders-20261016.log`:330 generated variants plus native
  matrix/physical expert-pointer validation pass; exits before device creation.

No GPU/model/inference/numerical/speed/recovery/sudo/service operation,
installation, API/frontend or release action occurred. Goal stays active for
CPU-only work; whole rewrite and physical acceptance remain incomplete.
Required Vashti changes:none.


## Scoped verified-row draft consumer (2026-10-16)

Source `1f46f766a13b405c740657e87ef12f843dac3d1a` connects an original MTP
consumer before explicit target commit acknowledgement. Delivery source
`0bf0b5f` then aligns emission with pinned serving. `PrefixPhase::Verified`
observes real provisional output/fault rows and selects n; the original host lease
is dropped. Original prefix commit is submitted, then `PrefixPhase::Pending`
provides `VerifiedCarryContext`, an immutable factory limited to the same original
R sources/window. Borrow requires the original opaque receipt, current row
generation, original graph and arena root, full proposed span and accepted n.
It does not require or fabricate committed target state. The existing committed
borrow continues requiring its real ready cursor. Both give the same original
`DeviceCarry` whose unknown copy completion retires the original context.

The actual global controller commits request-local sampler input history once,
sets coupled next-round controls and calls the original device-front/window MTP
consumer under this scope. Original full target/GDN/QSA/PLE transactions remain
owned. Actual MTP packet/fault completion is still required to release R. Original
commit join, second fault/I/O proof and all-family acceptance still precede target
cursor publication. Delivery source `0bf0b5f` emits genuine verified predictions
after original row/fault observation and commit submission, before next drafting.
Output delivery never grants completed state or final request success; later
commit/draft failure ends the stream with partial verified output. Original
delivery errors survive terminal cleanup. Failed observers take the original
terminal poison/close path, without granting completed target/model state.

A completed `PreparedDraft` then carries bounded wiping inputs, next target origin,
actual original MTP binding/cursor, first candidate and f64 joint probability to
the next loop. Confidence trimming never rewinds that cursor or loses the first
candidate needed by suffix policy. Only prompt seeding uses the earlier committed
carry API. State acceptance and EOS/cap output clipping remain separate; terminal
rounds skip next drafting. No new GPU data root, foreign engine or persistent
content is added. This is the selected path, not a default-off optimization.

Still-open coupled substitutions:

- Global graph push/copy/flush/replay/queue guards still complete prior work.
  A pending observer alone does not establish physical overlap. Establish an
  original source-readiness edge before commit, select the MTP lane before token/
  scratch/front preparation, and resolve actual buffer and whole INFO-copy hazards.
- Pinned `generate.cpp:11089..11140` submits commit, emits accepted predictions
  and then drafts; final `wait_commit` is at11169. Controller source `0bf0b5f`
  now uses emit-before-draft order without claiming state readiness. Independent
  queue execution/final-fence fidelity are still unfinished. MTP observation and
  target final acceptance remain synchronous under the current global guards.
- Shared MAIN `completed_fence` credits all earlier context work, and shared I/O
  checkpoint recycles through a global stream flush. That assumption currently
  holds under global joins. Independent-lane work must retain unrelated pending
  allocation/retirement debt and original ownership; a peer fence cannot credit
  the target merely because its own consumer completed. Resolve this with the
  queue/INFO changes, not by deleting waits alone.

CPU/offline receipts under `/media/jmicko/faststore/kuno-development`:

- `strata-pending-draft-test-build2-20261016.log` and
  `strata-pending-draft-cpu-20261016.log`: scoped producer builds;113 focused CPU
  cases pass, including all5 original carry tests and5 controller tests. The new
  carry case distinguishes verified row readiness from committed target state,
  rejecting stale generations/bindings/windows/prefixes. The producer case checks
  actual completed MTP identity/cursor and confidence-trimmed token metadata.
- `strata-draft-delivery-test-build-20261016.log` and
  `strata-draft-delivery-cpu-20261016.log`: final source above builds and all113
  focused cases still pass across13 nonempty suites. These include accepted-prefix
  rejection, EOS/cap independence, sampler, original lease, GDN/QSA/PLE, INFO,
  queue, flight and retained parameter-program checks. They do not execute the
  live whole-model observer or prove numerical/physical behavior.
- `strata-pending-draft-{core,off}-clippy-20261016.log` and final
  `strata-draft-delivery-{core,off}-clippy-20261016.log`: selected/off strict
  locked release Clippy pass. Changed-source formatting and `git diff --check`
  pass. Cargo still emits its existing duplicate-main-bin manifest warning;
  there are no Rust/Clippy warnings.
- `strata-pending-draft-build-20261016.out`: immutable initial scoped source
  `1f46f76`, SHA256
  `cd4c52ba865d6d7094f3b5326966e0037078556f5f4dab256b940fd7ce467a69`.
- `strata-draft-delivery-build-20261016.out`: immutable final source
  `0bf0b5f9fee0085b90faf9467855aabd8a0e2a90`, SHA256
  `f9a01ca8e774662bc1a0e223fc251ba5f27ae8af1509d7ce9c37914df37538f4`.
- `strata-draft-delivery-shaders-20261016.log`: final artifact validates330
  generated variants plus cooperative-matrix/physical-pointer paths. The early
  validator exits before adapter/device creation. No GPU resources were used.

Full source audit, arbitrary-input/context/placement substitutions, historical
core-regression failures and numerical/physical/recovery gates remain open.
No throughput result. No GPU/model/sudo/service/API/UI/install/release action.
Required Vashti changes:none. Goal remains active for CPU-only work.


## Original pending commit retirement (2026-10-16)

Source `543bc73e078ee147329d4d73ea871a54bbf11bc9` connects original pending target submission
and shared MAIN retention as one private `PendingNativeCommit`. The selected
multirow path acquires it immediately after real commit submission and before the
scoped MTP observer, then consumes it through its original runtime join. Count1
has no extra submission or hold. There is no caller-declared completion, clone,
send, reset, serialized identity or additional GPU data root.

`MainAdmission::completed_fence` still validates real observed MAIN growth and
limits. While original pending debt is held, it cannot lower the upper allowance
on behalf of an unrelated completed consumer. Up to two holds fit the native lane
bound; admission itself is not lane authentication. Original runtime graph/scope/
submission/fence validation remains mandatory. Only a successful exact original
join may release that guard's hold. Release alone never resets telemetry/credit;
a later checked completed boundary may lower the allowance only with no holds.
Duplicate/stale/foreign joins fail in the original runtime before ledger release.
No failed or unexpected-growth observation becomes credit.

An unjoined/failed guard atomically retires its original context/sequence, retaining
debt without taking admission's mutex. Its completion field drops before its
retained original client. Existing model close and checked transient cleanup still
attempt actual drains for retired contexts but cannot claim clean. Successful
join does not clear prior poison. Existing family/fault/I/O proofs remain required
before target state/cursor publication; verified emission remains separate.

This resolves the pending target's peer-fence credit prerequisite, **not physical
independent execution**. Global graph/stream/INFO/recycle waits remain. Any future
independently pending unrepresented work must own its corresponding retention;
this guard is not permission for arbitrary peer fences to refund all context work.
Next retain the original recording's actual access intervals under its existing
metadata bound, protect full-owner INFO copies, and establish precommit R readiness
before moving all MTP staging/math/observation to its independent lane.

CPU receipts under `/media/jmicko/faststore/kuno-development`:

- `strata-pending-retirement-final-test-build-20261016.log`: locked selected release
  test build passes. `strata-pending-retirement-cpu-20261016.log`:152 cases pass
  across16 nonempty suites (previous113,13 MAIN,25 executor admission,1 pending
  completion/drop). New checks cover both bounded holds, peer fence and allocation
  sequences, delayed refund, unexpected growth/poison and monotonic failed Drop.
  They test actual host accounting helpers, not Vulkan physical completion.
- `strata-pending-retirement-{core,off}-clippy-20261016.log`: strict selected/off
  locked release Clippy pass; changed-source format and `git diff --check` pass.
  Existing duplicate-main-bin manifest warning remains; no Rust/Clippy warning.

No shader/binding layout changed, so the unchanged shader suite was not repeated.
No new immutable artifact was frozen for this host-accounting block. The latest
frozen `0bf0b5f` artifact and330 offline variants are historical; it does not include
this change. Build current source for future protected whole-model execution.
No GPU/model/sudo/service/API/UI/install/release action. Required Vashti changes:none.
Whole source audit, arbitrary-input/context/placement substitutions, historical
core-regression failures and numerical/physical/recovery gates remain open. No
throughput result; goal stays active for CPU-only work.


## Original recording access and INFO copy ownership (2026-10-16)

Source `a8e538f52767fde9a211cf9ffea2a8a9d2f9c3f5` retains canonical actual
accesses in `strata_schedule::Plan`, derived from the existing planner's original
node histories. They include real descriptor views, physical read dependencies,
copy/fill and indirect reads supplied by `Graph::node_ranges`. Shared readers do
not conflict; any historical write remains a write. Canonicalization never fills
unused holes or narrows a descriptor. Sorted disjoint ranges support bounded
binary lookup without comparing every new copy to every original dispatch.

The actual original recording owns these ranges on cold capture and direct replay.
`Commands::metadata_bytes` includes the retained vector's actual capacity; the
existing64MiB retained-recording limit stays enforced. Temporary interval counts/
capacities retain their existing bounds, with contraction after adjacent-range
compression. No new GPU root, source mirror or content-bearing cache is added.

`strata_info_transfer::accesses` resolves the same complete original Vulkan
source/target owners and usage/span/alias checks used by the actual copy command.
A new source is a full-owner read and destination a full-owner write. Record
positions cannot pretend a whole INFO page copy writes a narrower view. Temporary
access capacity is included in the original2MiB INFO host-bound assertion and is
dropped before submit's validation allocation. The native CPU harness aliases the
actual sibling schedule module so both unmodified source modules are tested;
there is no mock replacement for their ownership/layout logic.

`Graph::upload_parameters` now preflights both actual original flight recordings
and rejects partial/missing/foreign-family owners before any fence credit. It uses
the retained original ranges to join only conflicting original lanes through
`complete_lane_submission`; that function still requires the actual matching
Vulkan fence and exact recording identity. Nonconflicting math remains owned and
gets no completion credit. Only the old INFO copy's own fence is required before
resetting its copy pool. Original exported completion remains retained until ack;
`543bc73` still protects pending target MAIN debt from peer observations.

A newer disjoint INFO upload can coexist with older math. Final drain therefore
decides unconsumed INFO visibility AFTER joining original math and crediting each
actual header epoch. An older consumer cannot excuse the newer copy's handoff.
Unknown/partial failure retains original resources; cleanup/erasure is unchanged.

CPU receipts under `/media/jmicko/faststore/kuno-development`:

- `strata-pending-access-final-test-build-20261016.log`: locked selected release
  test build passes. `strata-pending-access-cpu-20261016.log`:156 tests pass across
  the16 nonempty focused suites (previous152 plus3 original schedule and1 INFO).
  New checks cover historical reads/writes, gaps/boundaries, malformed tails,
  whole-page INFO writes, compact reserved capacity and4096 seeded comparisons
  against the original descriptor lists across compute/copy nodes. These are host
  range/ownership/epoch tests, not live Vulkan visibility, queue or erasure proof.
- `strata-pending-access-{core,off}-clippy-20261016.log`: strict selected/off locked
  release Clippy pass. Changed-source formatting and `git diff --check` pass.
  Existing duplicate-main-bin Cargo warning remains; no Rust/Clippy warning.

No shader/binding layout changed; the unchanged shader suite was not repeated.
No new immutable artifact was frozen. Frozen `0bf0b5f` predates both retirement and
access ownership changes; build current source before future protected execution.
Global graph push/stream flush/replay/queue-switch/recycle waits still serialize
execution, so this is not physical overlap or full scheduling parity. No speed
result. No GPU/model/service/API/UI/install/release or protected driver/inference
operation ran. Only user-authorized sudo timestamp validations ran in Demons.
Required Vashti changes:none. Whole rewrite and physical acceptance remain open.

### Next connected scope design

This remains the connected independent-execution contract. Source `d2d3228`
implements the original precommit R dependency, pinned source/read-only interval,
request scope and lane selection described in1/2 below. Existing global drains,
primary readback/reuse and hidden metadata/initialization boundaries still prevent
physical independence. Do not describe the scope API alone as completion. Follow
pinned serving's real order: verified output rows; submit accepted-prefix commit;
emit accepted predictions; draft on its own queue; wait for commit before final
session access. The current source already selects that logical consumer order.
Complete the dependencies as one connected scope before claiming independent math:

1. Establish an actual main-to-MTP verified-R readiness dependency BEFORE commit
   work is queued. Bind it to the original graph, active request scope, original
   model/source owner and monotonic backend identity. This is a queue dependency,
   never numeric/model-ready/erasure authority. The immutable original source
   grant/generation/window proof remains mandatory. Prevent new writes/reset of
   that source while the dependent consumer is live.
2. Enter the scoped MTP lane before sampler `begin_round`, `start_window`, scratch/
   token/carry preparation and `prepare_window_first`, not just at named capture.
   Keep unkeyed preparation and every named first/continuation branch on this lane.
   Do not enqueue a fresh main-queue readiness signal behind its pending commit.
3. Gate new original descriptor/copy/INFO accesses against the retained flight
   inventory. Preserve actual conflicts, original recording resources, copy
   timelines and `543bc73` retirement. Queue selection must distinguish queued
   nodes from already submitted flights (`Graph::pending` currently combines them).
   Do not globally delete all completion guards.
4. Cover observation and reuse, including each positive-confidence MTP cell.
   Source `7ad47ca` now routes selected native packet readback through its actual
   native copy/fence/HOST_READ/map path. Source `7cb1cbc` additionally removes
   selected native upload remaps/primary reuse transitions/clears, but its all-owner
   checkpoint still globally joins. Selective original-owner authority remains
   required. Ordinary and terminal cleanup readback retain their Wgpu path. They cannot silently join the target before all
   intended independent work or remap/overwrite live upload/INFO owners. Keep
   actual map/fault completion, source-copy completion and final audit/erasure.
   No fake per-lane successful fence or metadata-only reuse credit is allowed.
5. Scope success/failure/final close must retain original dependency, source,
   recording and pending-commit owners until their real joins. Unknown Drop may
   retire, never forget or declare clean. Preserve original target family/fault
   proofs and verified-output-versus-completed-state distinction. Cleanup must
   join known pending work even after the original context is retired.

Actual source findings to preserve rather than reimplement unnecessarily:

- Private uploads already use `MAP_WRITE|COPY_SRC` mapped-at-creation owners and
  native copy nodes, not ordinary `Queue::write_buffer` staging. Inspect
  `transient_io.rs` and `stream.rs::push_transfer` before inventing a new uploader.
- `batched_uniform` already gives named branches their original pinned parameter
  pages separately from unkeyed `core_info_pages`. `prepare_native_info` unmaps
  full original sources, resets the slab/cursor and returns full-owner copies.
  Keep these owners; prove actual hazards instead of assuming record offsets or
  replacing all INFO storage. Upload-source CPU remapping also requires genuine
  completion/reuse authority before the copy, not just a later read-range check.
- Retained metadata inspection exists in `server.rs::retained_resource_metadata`
  without submitting/cleaning. `resident_util::resources` selects it under its
  existing policy; native global roots already require it. Check the actual
  selected policy and MTP physical guards so metadata calls do not restore a
  hidden global wait while the independent scope is meant to be active.

Continue full source/ownership integration, not timing trials. Arbitrary-input,
context/embedding placement, historical regression triage and final numerical/
physical/recovery acceptance still remain beyond this scheduling scope.


## Original precommit R readiness and draft lane scope (2026-10-16)

Scope source `dcd783b66e09100da407df8509378a2d4ecaa6c3` (scope implementation
`d2d322820c8861c691bcc0a3b7efb91c6f5d8b30`, terminal-drain follow-up `dcd783b`)
connects the selected verifier to an original
request-owned draft scope BEFORE accepted-prefix commit is queued. The original
row/fault observation and lease release still precede preparation. The caller
supplies `gr::full_private` for the same current residual owner used by the real
row output. The existing carry receipt additionally authenticates generation,
original graph/model binding, verified window and selected prefix at consumption;
the new queue token cannot replace that proof or publish target state.

Internal runtime methods (not HTTP/application APIs):

- `prepare_native_strata_draft_scope(source: Handle)` retains the original full
  same-stream allocator binding AND Wgpu resource. The native graph requires the
  original initialized STORAGE/COPY_SRC owner, joined MAIN and no existing scope.
  It installs retention before fallible synchronization. A dedicated reusable
  `strata_handoff::Handoff` actually submits main signal and draft wait now, before
  the commit. Only both successful submissions arm the scope. The original
  backend identity is `(graph, monotonic scope generation, request nonce)`;
  metadata alone cannot create a live dependency. No debug/serialization added.
- `enter_native_strata_draft_scope(original token)` validates the exact original
  graph/request/generation, rejects unsealed parameter work and enters before
  the entire Pending observer. Thus sampling controls, scratch/token/R-copy/front
  preparation and named first/continuation branches all select the draft lane.
  `select_branch_queue(None)` keeps that lane until scope end. Target branches
  inside an active draft, premature draft branches and repeated entry fail.
  The MAIN->draft transition consumes the actual precommit dependency, without
  inserting a fresh readiness signal behind its pending commit.
- `end_native_strata_draft_scope(original token)` flushes and joins actual
  consumers/readiness, restores the real MAIN queue visibility and only then
  drops the retained original source. It does not accept target state, numerical
  fault or erasure. The separate original pending-commit join and all-family/
  PLE/fault/I/O proof still precede target cursor publication.

`strata_draft_scope::Book` stores only bounded ownership metadata. The original
source's actual interval is read-only from preparation through scope completion,
including partially submitted dependencies. Original node descriptor/physical/
copy ranges and full-owner INFO transfers reject overlapping writes. Direct
replays check their original retained actual access plan too, so skipping enqueue
cannot bypass source protection. Shared readers, disjoint intervals and other
buffers remain legal. No new GPU data root, weight mirror or persisted content.

The private non-sendable `NativeDraftScope` owns the same original client and
retirement flags. Its failure guard exists before backend preparation. Unknown
Drop only poisons/retires, without locks, joins or source release. Backend source/
allocator/synchronization owners remain retained through actual owning terminal
drains, including failed scopes. Terminal recording cleanup discards scope
metadata only after actual queue/readiness completion. Follow-up `dcd783b` also
requires the original ordinary cross-lane handoff's completion in this terminal
path: a failed flush followed by empty math must not bypass its unknown fence. Unknown Graph Drop
retains the allocator binding and synchronization object as well as its existing
recordings/resources until protected process/device teardown. It cannot mint
state readiness or make failed inference successful. Existing pending target
MAIN debt remains held until its exact original exported join.

**Still serialized:** scope entry, node push/transfer, graph execute/replay, stream
flush, metadata, confidence readback and I/O recycle still contain global/primary
boundaries. In particular scope entry may physically join the target before
verified delivery even though the original exported acknowledgement is retained.
This block is source/dependency progress, NOT physical overlap or source schedule
parity. Do not remove those waits independently. Next connect exact current-lane/
conflicting-flight admission, full INFO SOURCE remap authorization, original
confidence packet copy/map and selective upload/readback reuse together. Handle
`Graph::pending()`'s queued-node versus submitted-flight distinction and any fresh
Wgpu initialization/primary handoff. Positive-confidence multi-cell observations
must not silently reintroduce a primary wait behind target commit.

CPU receipts under `/media/jmicko/faststore/kuno-development`:

- `strata-draft-scope-final3-test-build-20261016.log`: locked selected release
  test build passes for current source including the terminal-drain fix. The
  earlier `final2` build passed before that fix. The first and `final` (without
  a numeric suffix) build logs record corrected mutability/constant compile
  errors; those two are not successful receipts.
- `strata-draft-scope-cpu-20261016.log`:163 tests pass across19 nonempty CPU
  suites: previous156 plus3 original draft scope cases, the existing handoff
  epoch case and3 existing queue-topology cases. Scope cases cover dependency/
  entry/finish order, foreign/stale/repeated IDs, exhaustion and terminal drain,
  preentry target versus active draft/unkeyed lanes, and exact source interval
  read/write boundaries including partial-dependency retention. These are host
  state/range checks, NOT actual Vulkan submission, visibility or completion.
- `strata-draft-scope-{core,off}-clippy-20261016.log`: selected/off locked release
  strict Clippy pass. Changed Rust formatting and `git diff --check` pass. Only
  the existing duplicate-main-bin Cargo manifest warning remains.

No shader/binding layout changed; unchanged shader checks were not repeated.
No new immutable ordinary release artifact was frozen; latest frozen `0bf0b5f`
is stale for this and prior retirement/access changes. Build current source before
future protected execution. No GPU/model/service/protected inference/driver/
API/UI/install/release action or speed result. Sudo timestamp refresh alone
succeeded in the shared Demons shell. Required Vashti changes:none. The complete
rewrite and arbitrary-input/context/placement/regression/numerical/physical/
recovery gates remain open; goal stays active for CPU-only work.


## Original native packet readback (2026-10-16)

Source `7ad47ca4427c4a2e97cb22b501a34bbd9996cb16` connects the selected native engine's
private packet observations to the original native execution plan. This covers
normal target, fault, MTP/confidence and sampling packet reads selected through
`read_private_resources`; it is not a change to HTTP APIs or model arithmetic.
Ordinary feature-off and dedicated terminal cleanup reads keep their original
Wgpu implementation. Required Vashti changes:none.

- `TransientIo::native_readback` keeps the original tracked Readback allocation,
  budget and original source `copy_bounds`. It enqueues a complete actual owner
  fill (including padding), then the complete original aligned source copy.
  Returned logical bytes are not permission to narrow the actual copy. All
  allocation/lease duties exist before either native enqueue can fail.
- `Graph::execute_readback` accepts only real copy/fill nodes and uses an unnamed
  original recording. It cannot replace an open named math branch's retained key
  or signature, or mark packet I/O as another submitted math invocation. Its
  existing selected compute/copy timeline plan owns both original source and
  destination through the actual final lane fence, including all copy segments.
- Actual native copy/fill recording now emits TRANSFER_WRITE -> HOST_READ buffer
  availability for MAP_READ spans. A real queue fence alone is not substituted
  for that memory dependency. Full initialization precedes raw host access;
  Wgpu's initialization tracker is not used to invent native output writes.
- `strata_readback::read_joined` borrows the original Wgpu allocation's Vulkan HAL
  device/buffer and allocator mapping synchronously AFTER the genuine copy fence.
  It invalidates the full mapped range when non-coherent using the HAL's actual
  atom alignment. A local guard unmaps on success/error/panic; no raw pointer or
  HAL guard crosses an async boundary. Failed maps/unmap/late completion retire
  the original I/O owner. One original30-second host-map batch deadline covers
  allocation copying and final unmap acceptance; it never restarts per packet.
- Only the original logical bytes are copied into `PrivateHostAllocation` and
  the existing wiping `ReadbackController`. The same original lease stays held
  through returned Bytes/future destruction. Native owners never call the Wgpu
  map/callback/unmap path; the local native map guard owns that duty instead.
  Terminal full erasure/audit and held-lease rejection remain unchanged. No new
  untracked GPU buffer, persistent content or weight/source mirror is introduced.

Cold inference source initialization is preserved through the actual existing
readonly Wgpu storage-root initializer, followed by its checked completion before
native copying. Actual inference sources require STORAGE/COPY_SRC and original
source/backing bounds. Opaque private I/O/external staging owners are not inference
source bindings. Scoped verified R was already initialized before its pin and
cannot be silently reinitialized/reset. A cold source may still require primary
work; do not remove it to manufacture an independent cold-run claim.

The selected native packet path issues no primary Wgpu readback copy and requests
no Wgpu map callback or ordinary map poll. Its legacy `inline_readback_batches`
counter therefore does not receive fabricated callback credit; existing scoped
setup/flush/map counters still describe actual work. Those numeric counters are
not completion/numerical/erasure authority.

**Remaining at packet source `7ad47ca` (native remaps/primary reuse superseded by `7cb1cbc` below):** current graph push/execute/flush and scope-entry drains still join
all original flights. Shared `recycle_reuse` still remaps uploads via Wgpu and
submits primary transitions/readback clears. This block is not physical independent
execution or a throughput result. Next connect current-lane/conflicting-flight
admission AND original selective I/O reuse/source remap authority together. Keep
fresh source initialization, original commit exports, INFO copy/header epochs,
MAIN retention and terminal failure drains. Do not simply delete global waits.
Positive-confidence per-cell readback is now native, but its surrounding shared
checkpoint must also stop joining unrelated target work before claiming parity.

CPU receipts under `/media/jmicko/faststore/kuno-development`:

- `strata-native-readback-final-test-build-20261016.log`: locked selected release
  test build passes for final source. Earlier unsuffixed build passed before the
  final unnamed-recording/cold-source/deadline refinements; it is historical.
- `strata-native-readback-cpu-20261016.log`: 165 tests across 20 nonempty suites
  pass (previous 163 plus 2 original native readback plan/copy checks). New tests
  cover full original copy/owner limits, alignment/overflow/logical extents,
  exact byte preservation, no padding return, mismatched full spans and a single
  original deadline including late acceptance. They exercise real wiping host
  allocations and the production copy helper, not Vulkan mapping or visibility.
- `strata-native-readback-{core,off}-clippy-20261016.log`: strict locked selected/
  feature-off release Clippy pass. Changed Rust formatting and `git diff --check`
  pass. Existing duplicate-main-bin Cargo manifest warning remains.

No shader/binding layout changed; unchanged offline shader checks were not
repeated. No new immutable ordinary release was frozen; frozen `0bf0b5f` remains
stale and must not be used as current source for future protected execution.
No GPU/model/service/protected driver/inference/API/UI/install/release action.
Sudo timestamp refresh alone succeeded in Demons. Actual coherent/non-coherent
Vulkan map/barrier/fence/reuse/failure/erasure proof and whole-model numerical/
completion/recovery remain pending, alongside the other full rewrite gates.
Goal stays active for CPU-only work. Full rewrite and speed gate remain open.


## Original native upload and joined I/O reuse (2026-10-16)

Source `7cb1cbc38de0310a3f8a98f8acd874839c9bca32` connects actual INFO/data renderers and
native I/O checkpoints to the same original tracked upload owners. It replaces
`upload_mapped: bool` with `strata_upload::State`:

| State | Actual responsibility |
| --- | --- |
| `Idle` | CPU write permission revoked; actual queued readers/copies may still use it. |
| `WgpuMapped` | Original mapped-at-creation or checked legacy map; retain real Wgpu unmap duty. |
| `NativeReady` | Original joined checkpoint granted native write reuse; Wgpu remains unmapped. |
| `NativeWriting` | Grant consumed before fallible native map/render/flush/unmap; cannot stage or recycle. |
| `NativeWritten` | Native render/flush/unmap succeeded; same unsubmitted append page may receive more writes. |

The state is bookkeeping, not fence/numerical/erasure authority. Only the original
serialized stream's real completed I/O boundary grants reuse. A failed or partial
writer cannot become staged input merely because another fence completed.

`write_upload` now serves native/ordinary data uploads, fresh/appended INFO and
direct retained INFO replay. Producers receive bounded write operations only;
neither Wgpu's write-only contract nor original native ownership permits exposing
a readable mapped slice/raw pointer. `render_bounded` validates every exact
full-owner write including integer overflow and stops later writes after failure.
Initial source fill still zeros the full original API span including padding.
Actual original runtime controls/template renderers and byte encodings stay intact.
INFO slabs retain the original upload index beside the original target index;
append writes do not search/reconstruct or substitute another source owner.

First-use Wgpu mappings retain their actual view/unmap. Reused native sources use
a synchronous borrow of the original allocator's Vulkan mapping. The local guard
unmaps even on error/panic, and non-coherent full-range writes are flushed before
unmap, including partial/error writes. Only success advances Writing -> Written.
`seal_upload` requires an actually staged source, completes its actual Wgpu unmap
when needed, then revokes CPU permission before the GPU copy. Native sources were
already flushed/unmapped. The source bytes/owner are never replaced by metadata.
Terminal erasure resets only bookkeeping and keeps the original real remap/
volatile full wipe/flush/audit; dirty reuse grants no storage release or clean
receipt. No new GPU data root, content persistence or weight mirror.

The selected stream's `recycle_transient_io` retains the real flush, then calls
`Graph::join_all_io_sources`. That method joins original math/copy/INFO/readiness/
handoff fences and rejects unsubmitted nodes, live flights or transfer owners.
Only then may `recycle_native_after_all_joined` validate the original request,
zero held leases, empty pending INFO/slab and accepted GPU errors. Its real all-
owner completion preflights the ENTIRE upload cohort before granting any writer.
The intrusive free index accepts only actual Wgpu-mapped/native-ready Uploads;
writing/staged uploads reject. Original INFO pins, charges, retained recordings,
exports and pending MAIN debt stay owned. No Wgpu upload map callback or primary
reuse transition/Readback clear is requested by this selected checkpoint. Native
readback's actual full fill/copy/HOST_READ/fence path remains required before
its next host observation. Legacy feature-off reuse keeps its old callback path.

**Still a global boundary:** this is not selective independent reuse or physical
overlap. Do NOT delete the global joins and call the all-owner unsafe function
after only a peer/current-lane fence. The next connected block must obtain the
original I/O owner inventory, check full host-write reuse accesses against original
retained flight ranges AND current INFO copy ownership, and exclude every still-
used owner from native write grants/free-index publication. Coordinate that with
scope entry, push/transfer, execute/replay and stream flush current-lane/conflict
joins; preserve queued-versus-submitted work, real INFO epochs/visibility, cold
Wgpu initialization and original final/failure drains. Audit retained replay's
actual source/descriptor pins as well as live flights; a future replay must not
silently consume an overwritten captured upload. Keep positive-confidence per-
cell checkpoints under that same original scope. No timing result is used here.

CPU receipts under `/media/jmicko/faststore/kuno-development`:

- `strata-native-upload-final-test-build-20261016.log`: locked selected release
  test build passes with the complete connected upload/checkpoint/owner harness.
  Earlier unsuffixed build predates checkpoint connection and reported unused
  NativeReady/after_joined; it is not the final warning-free receipt.
- `strata-native-upload-cpu-20261016.log`: 276 cases pass across 22 nonempty suites:
  previous 165, 2 original native upload state/bounded-writer cases and 109 original
  I/O/cache/parameter owner cases. New cohort test rejects partial/staged tails
  before any writer publication and checks the actual intrusive index's native
  grant consumption. Native writer cases check bit-preserving/append/bounds/
  overflow behavior and no later writes after failure. Existing lease/error/
  erasure/cache/index policies now compile/run against the real state refactor.
- Cargo rejected direct dependency-library test attempts because that package
  is outside this workspace and requires dev dependencies. Those two failed
  `strata-native-upload-library*-test-build` logs are not passed tests. Instead,
  `strata_io_owner_tests` includes the actual original I/O, parameter and cache
  files. Only its device-activation predicate is disabled; no queue/fence is
  simulated. The included I/O source uses the lab root's feature namespace, so
  its shared-budget cfg is generic, not the production dependency's selected
  shared-budget variant. This harness proves host logic/type/policy cases, NOT
  native activation, actual selected device limits, map/coherence or erasure.
- `strata-native-upload-{core,off}-clippy-20261016.log`: locked selected/off strict
  release Clippy pass. Changed Rust formatting and `git diff --check` pass.
  Existing duplicate-main-bin Cargo manifest warning only.

No shader/layout change or new immutable ordinary artifact; frozen `0bf0b5f`
remains stale for current source. No GPU/model/service/protected inference/driver/
API/UI/install/release work. Sudo timestamp refresh succeeded in Demons. Required
Vashti changes:none. Native coherent/non-coherent write/flush/unmap/reuse, real
queue/barrier/failure/erasure and whole-model completion/numerical/recovery gates
remain pending, alongside other full-rewrite obligations. Goal stays active for
CPU-only work. No physical independence, source parity or speed result claimed.
