# Complete Strata Intel inference rewrite: completion gate

Approved 2026-10-14. This is backend engine work, not gateway/API/UI/release work.
The user requires a complete coherent Rust port before live speed testing. This
checklist is a source-to-source audit, not a sequence of throughput experiments.
Each row needs its connected consumer, real ownership/placement and state contract;
adding an unused helper or optional switch does not complete a row.

**Current CPU-only source:** `e5981a138a58a4209448d8091b1f576c3b1968b4`, branch
`codex/strata-rust-port`. Read the [connected native QSA indexer receipt](kuno-strata-rewrite-completion.md#native-qsa-window-indexer-and-direct-pooled-scoring-2026-10-17).
Actual decode now consumes the native 256-thread serial window indexer, completed
pooled keys in the original persistent owner and direct causal scoring. The
per-query retained-prefix copy, separate mean/norm/publication path and two unused
pooling-buffer clears are no longer selected. Original tail/spare semantics and
accepted-prefix cache/pooled suffix handling remain connected. Original ownership,
full-request erase, source-root retention and conservative admission are preserved.
No new GPU root, model copy, CPU content readback or public API/configuration.

581 distinct selected CPU cases/80 nonempty suites, strict core/off binary lint,
changed formatting and whitespace pass. Stable-source frozen artifact SHA256
`1254d3ba1cc8fa9107f8c99060dd07c5aa45ac3aab6d4fe892c8ce7f40cceb01`, independently source/ordinary/frozen/mode 0500/default-features
verified. Only its early offline branch ran: 371 shaders/139 physical-pointer
checks/five rounded-product modules/32 matrix configurations pass. These prove
CPU/compiler acceptance, not physical numerical/model/recovery/throughput parity.
Earlier original artifact/Q8/tokenizer/MTP/shared/routed/GDN receipts still apply.

**Whole rewrite incomplete; speed gate closed.** New source substitution keeps
at most three pre-window FP16 keys in Workgroup storage from the existing packed
cache, versus upstream's separate persistent tail/snapshot. The spare is still
in the existing final pooled slot. Native 256-thread order, math and original
buffer dependencies are connected; Vulkan math and physical replay/state handling
remain unverified. Continue QSA projection/norm/KV/selection/attention and source
placement/data-transfer audit, then complete remaining whole-branch ownership/
math/prompt/sampling/queue/pricing/substitution checks. Protected count-one/
multirow/context/retained/hot-request and failure/all-resource recovery acceptance
must precede speed. CLI compiled-public/one-request; private caller separately
gated. Goal active CPU-only; authorized sudo refresh only. No GPU/model inference,
service/install/release/frontend action. Vashti changes none. Canonical checkout
and inherited untracked cache preserved.


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
The final audit also corrects usage accounting: NO_HOST skips the only callback
that increments expert usage. Reference counters remain zero, so the every4-window
host job proposes no promotions. GPU-derived usage is an explicit extension,
not a prerequisite of the selected measured path. See the latest receipt below.
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
| Artifact geometry, packing and tokenizer | `sycl/src/core/weights.cpp`, `src/io`, native expert layout, program model setup | Rust GGUF/tokenizer/binding exist. Audit format-specific load/conversion, chat-template and sampling behavior; original files remain shared/read-only. Source71dceafc and its original-artifact receipt verify actual two-shard binding, original target/MTP role identity, all nine full runtime-Q8 packed matrix hashes, once-folded norms and actual-vocabulary tokenizer controls. Raw IDs/rendered text remain the engine input; chat serving/templates are separate. |
| Exact dense placement and expert budget | `sycl/src/program/generate.cpp` model/cache setup; `sycl/src/core/expert_cache.cpp::open_sized` | Exact dense roots exist. Native global byte-prefix planning and an owning bounded GPU loader now compile; graph validation consumes the real artifact plan. Experimental core construction now invokes the sole global owner before prefill, and the core48-layer coordinator selects global routing/HOST fetch/role execution. Physical/numerical acceptance remains unfinished. All777 required original float tensors already match the selected native format (484 BF16,292 F32,1 F16); generic conversion is not required for this artifact. Source `89da25f` moves the shared original IQ4_XS embedding into the sole mapped HOST owner and updates constructor/reader/close accounting. Source5b520bb replaces the tiled embedding dispatch with original mapped-pointer gather for target and shared MTP, using original32-lane IQ4_XS work sharing. Physical/numerical acceptance and final source audit remain open. Sources `cbf4381`/`910cb58` replace the blanket5GiB allowance with a complete original dense inventory, preload/seal it before expert allocation, and size the automatic ranked prefix from actual remaining shared MAIN. See the dense-inventory receipt; physical placement and whole-device pricing acceptance remain open. The ordinary feature-off engine retains uniform banks. |
| Low-RAM host complement and promotion | `generate.cpp` source/adaptive tier; `expert_source.cpp` callback; `verify.cpp` NO_HOST | Original HOST complement, source ownership and joined physical publication remain connected. Source `3d6a0390` corrects selected NO_HOST to zero callback-derived counters and an actual bounded empty host job beside commit/MTP. No unused GPU counter readback, physical copy, registry revocation or unchanged-map upload is added at that tick. Prior `f9c7471f` authenticated promotion/copy worker remains explicit `strata-gpu-adaptive-usage`; ordinary selected mode uses4MiB adaptive HOST allowance versus12MiB for the extension. Original full request erase/rebind audits remain. Physical/numerical acceptance and final pricing audit remain open; do not duplicate the entire model in RAM. |
| GPU route/group planning and original HOST access | `native_router.dp.cpp::route<512>` / `verify_kernels.dp.cpp::resident_plan_kernel`, selected SYCL `DEVICE_PLAN=1,NO_HOST=1` | Source `0d9e3a52` replaces fused eight-token/serial planning with selected one256-thread block per token (one active32-lane subgroup) and a separate128-thread packed parallel plan scan. Both use the SAME original private map root and native graph dependency tracking. Stable top10, winning-lane normalization, first-occurrence groups and original member order remain connected to direct authentic VRAM/HOST role consumers. Cell-ID/fixed8 plan layout and Vulkan math are explicit backend substitutions, not bit-parity proof. No CPU expert callback or guessed expert IDs. CPU/compiler acceptance passes; protected numerical/whole-model acceptance remains required. |
| Native shared expert and fork/join | `verify.cpp::sh_stream_on/record_window`, `shared_expert.dp.cpp::shared_expert_multi`, native BF16 projection and `native_moe.dp.cpp` | Source `46e695a7` connects selected target shared compute fork, independent native Q8/fault roots, fused SwiGLU packing, original rank-one BF16 gate, vec4 sigmoid scale, ordered combined reduction and real all-peer target fence in one block. Actual cold/capture/replay Nodes, family pools, INFO edges and admission quotes are connected. CPU scheduling/shader interfaces pass; physical arithmetic/overlap, shared-to-MTP queue reuse and complete device/recovery acceptance remain open. See the connected shared-expert receipt below. |
| Native expert math and grouped rows | `sycl/src/kernels/cuda/iq_kernels.dp.cpp::row_entries`, `native_gu_kernel`, `native_down_kernel` | Packed integer math/BDA roots exist. Separate GU roles, 4/2/1 chunks, activation, intermediate storage and downstream consumers are now selected together in the Intel core; all 158 WGSL/SPIR-V variants validate, including 105 physical-pointer checks; numerical/model execution acceptance remains pending. The global core now uses one per-role launch containing pinned row_entries4/2/1 loops, scalar IQ4_XS, native signed integer finish and XOR reduction; numerical/model execution acceptance remains pending. Historical fixed-chunk kernels are retained separately. Source1a034311 connects native subgroup32 Q8_1 GU preparation and fused direct SwiGLU/Q8 DOWN over the same actual private PART/status/planes. Original four GU/two DOWN formats cover all144 installed routed tensors. Actual rounded products are pinned by the shared native NoContraction postpass. See latest source/frame-grouping substitution receipt; device numerics/whole-model acceptance remain open. |
| GR write/read/norm/down/up/injection | `sycl/src/kernels/cuda/fused_gr.dp.cpp::fused_gr_read_multi`; `verify.cpp::gr_read_group` | Native pending-write norm, sliced BF16 down/injection, fixed slice reduction and up/mix/commit are now connected and unconditionally selected in the core, including explicit PLE/final boundary writes. Resident original weights and the enlarged private scratch owner are charged together. Structural acceptance passes; numerical/model execution acceptance and final substitution audit remain pending. Source `37dced9` connects default full-prompt F1 row scales, packed BF16 norm/SiLU images, direct original projection consumers and residual-based mix recomputation together; see the packed-GR receipt. Sources `40fe97e` and the latest sole-residual receipt connect mixed images and prompt GR/PLE in-place storage, including actual MTP teacher scratch. Final substitution/numerical/device acceptance remains open. |
| GDN projection/conv/gates/recurrence | `verify.cpp`, `cuda/native_gdn*.dp.cpp`, `cuda/fused_gdn.dp.cpp`, `cuda/verify_kernels.dp.cpp`, `prefill/kernels.dp.cpp` | Core connects native BF16 alpha/beta, controls, recurrence/norm and accepted-prefix replay. Prompt scope selects contiguous H, causal convolution/history/L2, CB32 recurrence and directly consumed FP16 output. Source 9a7a23b connects prompt-capacity whole-chunk math without per-element status/replay owners, carved AB staging, and native row/value-head recurrent state shared by prompt/decode/repair. The large chunk caller/acceptance is connected in the latest controller receipt; physical execution and final numerical/queue audit remain open. Split verifier mode is an optional upstream path, not selected by the measured server. Source 39c8d9e1 connects default native 128-thread/four-column-tile accepted-prefix state commit over actual retained roots, replacing512-thread state-only output replay. See latest receipt; Source 7a0489d3 now connects authenticated verification-only deferral for recurrent/history writes, static replay mode and native history-only commit; see the latest receipt. Physical numerical/context/failure acceptance remains open. |
| QSA projection/norm/RoPE/KV/indexer | `verify.cpp`, `cuda/native_qsa*.dp.cpp`, `cuda/qsa*.dp.cpp` | Core now connects INT8 KV for current and committed rows, native fused norm/RoPE, FP16-rounded raw index keys, retained pooled blocks, native Intel scores, 64-cell attention/merge and rejection clearing together. The measured server explicitly selects INT8 for target and MTP; source2a3d1ad ports both together and the combined CPU shader/layout checks now pass. Numerical/GPU acceptance is pending. Earlier generic-default FP16 scope claims below are superseded. Numerical/state acceptance remains required. Source e5981a13 connects native 256-thread serial-window indexer and direct retained pooled scoring; separate per-query copying/mean/norm/publication and unused pooling clears are removed from decode. Original suffix acceptance and full ownership/erase remain. See latest source/substitution receipt; physical numerics and full QSA/placement audit remain open. |
| PLE source rows, projection and history | `verify.cpp`, `ngram`/`ple` kernels, PLE table reader | Core connects native BF16 MAX8 projections, normalization/gate/conv/history, deduplicated positional reads and accepted-prefix ownership. Prompt calls now consume a bounded, joined double-buffered source prefetch. The latest controller receipt connects native large chunk geometry, joined uploads, retained tail and exact history acceptance. Sources `595393e` and `57147a8` make actual decode and prompt history GPU-authoritative in existing OLD/NEXT roots, with checked CPU token-only publication, GPU accepted-prefix history and independent PLE capture roles. Default prompt/decode float readbacks and prompt diagnostic tail copies are removed; explicit diagnostics and the legacy CPU oracle remain. Source `131d29c` selects the native scoped accepted-prefix consumer before state/cursor publication; the diagnostic checkpoint path keeps the older schedule. Pending event/graph fidelity and numerical/device acceptance remain open. Prompt loans are disabled in the measured32768-context configuration and are not a completion prerequisite. Never persist request n-grams. |
| Prompt prefill and streamed experts | `sycl/src/prefill/prefill.cpp`, `gemm.dp.cpp`, Intel CMake build selection | Rounded BF16/FP16 GEMM, prompt GR/AB/head and original-source FP16 expert consumers connect to actual initial prompt calls. The bounded prompt path now uses logical-ID expert order, checked native transfer nodes, an eight-slot raw ring and two paired dequantization slots. The latest owned-consumer receipt connects full2048-row owners, non-overlapping raw/grouped route packets, causal math/state scopes, the unbatched final prompt input uses normal count1 decode for the first head, and full device MTP teaching covers the preceding prefix (up to8192 matrix rows). Independent transfer/compute queues and real release boundaries are connected. The measured server uses4096 prompt rows and32768 context. Source `bcc30dd` connects4096-row prompt owners/ABI; source `8d301b1` subsequently connects32768 runtime context; full-prefix/device acceptance is open. Source `12a60bf` connects native phase overlays, actual2080MiB admitted roots and256-query score/selection reuse; see the latest prompt-overlay receipt. Packed/in-place prompt scratch is connected in the latest receipts. Source allocation granularity and shared inventory are connected in the latest exact-page receipt. Physical/runtime/HOST pricing and lifetimes still require final review. Its no-borrow selection and nonresident fraction select the existing eight-slot/two-dequant-ring path; alternative loans/ring sizes are not prerequisites. Physical whole-model acceptance remains open. Pinned SYCL CMake does not build the CUDA/HIP MMQ/fused implementation. |
| Prompt dense/expert matrix backend | Intel oneMKL rounded BF16/FP16 GEMMs | Connected Rust cooperative8x16x16/F32 kernels now consume the actual rounded dense and grouped GU/DOWN images, with enabled device features, SPIR-V native registration and required16 subgroup in both pipeline paths. See the cooperative-matrix receipt; physical numerical/model acceptance remains open. |
| MTP loading/prefill/draft/head | `src/core/mtp.cpp` as built by Intel target, Intel kernels | Core connects MAX8 front, packed native INT8 cache, dense attention, BF16 FF/head GR and native scaled Q2/router/fused-shared consumers. Positive-threshold chains now keep hidden carries on device and observe the same token/probability/fault packet as zero-threshold chains; mapped/event scheduling remains open. First/grouped device-source and ids-only target result consumers now exist; the shared-context constructor receipt below supersedes the separate-client blocker. Core construction now selects the actual target client and one combined account. Source9de7fef pins the original target embedding/head once, shares them with MTP and gathers target embeddings on GPU. Actual device/state acceptance is pending. Source0bfbea8 connects the original-order CJK106299-row Q6 draft head, inverse-ID history mapping, compact confidence and coupled sampling; source2a3d1ad connects INT8 caches. Combined CPU checks pass, with full numerical/GPU execution still pending. Source `862f5fa` separates first-front staging and actual observation from capture, and prestages all coupled draft controls once per round. Source `8262367` connects typed MTP phases/4096 boundaries and original model-instance graph namespaces; source `d0670b0` connects direct original MTP replay and capture-derived host/sampler handback. Physical acceptance, commit/event fidelity, numerical/state acceptance and runtime policy remain required. See the latest device-end receipt. Source35f0990f connects actual shared-side fork, independent fault/scaling, original vec4/FMA combination and owning draft fence using the opposite existing compute queue. Its queue sharing with MAIN commit is an explicit backend substitution, not physical overlap proof. See the latest MTP shared receipt. |
| Output heads and sampling | `core/native_head.cpp`, `native_mmvq.dp.cpp`, `core/verify.cpp`, `core/mtp.cpp` | Source `ffe7293` connects original aligned wide Q6 A2/RPW1/SG32 and exact1–8 RHS math. Source `308de40` connects ONE whole-head target and compact-CJK MTP projection over the same original bounded roots, owning authenticated address tables, all physical read dependencies and exact admission. This supersedes the earlier per-page source gap. Final sampling/draw/verification audit and numerical/model acceptance remain open.  Source `8a4ee5b` connects original in-place coupled penalties, fixed 64 lists, request-K merge and original publication for BOTH draft domains; target sampling stays separate. Counter/history source audit below passes; numerical/device acceptance remains open. |
| Suffix proposal and draft policy | Intel speculation implementation and `DraftPolicy` | Sourcea26adc2 ports the native bounded trigram index, longest/recent suffix choice, cost priors, probing/stale rules and optional lookup-chain accounting into the selected controller. Configured MTP/target windows no longer shrink at the output cap; state commit and emission are separate. Coupled sampling and physical state acceptance still need end-to-end validation. Source `39c59c1` retains only fixed learned-policy bins in the actual loaded Session across completed requests, while suffix/token state still wipes; see retained-telemetry receipt below. |
| Target verification and accepted-prefix commit | `sycl/src/core/verify.cpp` checkpoint/commit methods | Source `131d29c` connects real provisional rows to accepted GDN/QSA/PLE candidates while original full-window transactions remain open, then publishes only n after real fault/I/O proof. Count1 uses its candidate and all-accept multirow runs original state replay. Old diagnostic full-window repair is not selected by the native controller. Source `4083258` connects the retained multirow commit graph with explicit dynamic accepted-count/position fields, original family preflight and count1 skip. Source `91e1f9e` connects original pending commit export/join; `1f46f76` connects a scoped original next-draft consumer before explicit commit acknowledgement. Source `0bf0b5f` emits real verified predictions before next drafting, matching the source delivery order. Source `f9d1e06` connects scoped admission/allocator lifetimes and INFO visibility; physical independent execution, full source substitution audit and independent numerical oracle remain open. Source `01291fe0` connects actual first final-prompt count1 Pending consumption to the MTP producer, before adaptation joins; source/scoped nonce receipt below. Full physical acceptance remains open. |
| Graph capture, queues and synchronization | `verify.cpp`, session/mtp graph construction, Intel queue/event implementation | Native Vulkan owned recording/replay now connects real compute/copy queues, concurrent buffer ownership, RAW/WAR/WAW timeline edges and joined completion. Actual descriptor/copy intervals and dequant-release signals are connected; Source `d40442c` connects direct retained target submission with typed control uploads and original family receipts; source `d0670b0` connects direct MTP invocation with original actual packet observations and sampler/actor handback. Source `b6732c8` adopts independent MTP compute and concurrent buffer families with semaphore handoffs. Source `414e2c3` connects actual current INFO copies on the native transfer timeline and avoids empty primary submissions; source `86f4c35` replaces the global single fence/last-recording assumption with actual lane fences and stable original recording flights. Source `131d29c` connects synchronous original accepted-prefix publication; source `4083258` connects the retained commit graph. Source `91e1f9e` connects original pending submission identities and exact joins; `1f46f76` connects scoped original MTP consumption before explicit commit acknowledgement. Real independent readiness/INFO/draft execution remain open; source `543bc73` couples original pending commits to retained shared MAIN debt; source `a8e538f` retains actual recording ranges and preflights whole INFO-copy conflicts; source `d2d3228` pins original verified R, submits its real readiness dependency before commit and scopes all unkeyed/named draft lane selection. Source `7ad47ca` connects original native packet fill/copy, real HOST_READ availability, final fence and checked HAL host map without primary Wgpu readback copies/callbacks. Source `7cb1cbc` connects native upload state/render/flush/unmap and all-owner joined I/O reuse without Wgpu remaps or primary reuse clears. Source `7477b66` connects original per-owner selective reuse, current INFO copy completion and named/future/live recording exclusions. Source `f9d1e06` connects scoped graph/stream current-lane/conflict admission, original allocator pins and copy-vs-math INFO visibility. Required ordinary/cold/final global drains remain. Physical independent execution and full draft/event fidelity remain unverified. Physical ring overlap, scratch lifetimes and full execution acceptance remain open. Measured serving lazily captures branches; eager all-branch warmup is not a prerequisite; a sequence of independent Wgpu launches is not equivalent.  Source `7c35203` moves both actual INFO upload owners to their corresponding compute lanes and all MTP copy/fill nodes onto MTP compute, preserving actual transfer access flags and per-lane fence/epoch debt. Scoped I/O reuse excludes pending peer INFO roots. This supersedes shared-transfer-queue control staging; no extra hardware queue. Host proposal/initial-window/session telemetry gaps and full physical acceptance remain open. Source `f9c7471f` replaces the borrowed transfer handle with one strong-device serialized queue authority used by BOTH actual native graph and adaptive copies. MAIN submits/joins and arena/TLS/registry stay on the sole engine thread. Full physical overlap/driver acceptance remains open. Source35f0990f generalizes shared nodes to authenticated active MTP work with opposite real compute-family pools, actual access admission, original all-peer waits/fence and recording ownership. MTP side work follows already submitted MAIN commit without a global CPU join. Physical independent execution remains open. |
| Request lifecycle and arbitrary input | upstream CLI/session/inference entry points | Source `dc7ace7` connects an owned raw-token request entry, original target/MTP construction, callback delivery and full checked cleanup; public diagnostics consume that actual path. Source `4ceb86f` connects completed target/MTP retained-state erasure with original-owner receipts before the current terminal close. Source `1f4b195560e5b8e7a83361f19c844a7104d11cbf` connects retained `Session::load/run/close`, bounded rearming from original erasure receipts, retained-ready owners and preserved policy/scope state; see the loaded-session receipt below. Sources `a8bbdf0`/`f1dd77f6a5a5cf98253488edec1fb761c1e9a0a4` connect original physical route-map restoration, live HOST-scope exclusion and full32K sampler request coordinates (4096 penalty ring unchanged). Hardware hot-reuse acceptance remains open. Source `e1458410b20f956586033b359c57186922d7ad8f` connects a bounded wiping reader/Rust-tokenizer consumer in the actual public entry. The executable is still compiled-public: an independently protected private process caller and verified hot request reuse and full device/failure/recovery acceptance remain required; no private/general service claim. No gateway/UI work required. Source `39c59c1` connects successful closed-request aggregate usage capture before full erase, exact original receipt-gated route/usage restoration and loaded-session policy wipe on close. Hardware hot reuse remains unverified. Source `01291fe0` adds an actual initial-window runtime nonce/full closing audit, distinct from generated request/reset scope authority; no fabricated nonce or closed-state receipt. |

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
   checkpoint still globally joins. Source `7477b66` adds actual original-owner
   selective grants/exclusions; graph/stream admission and allocator lifetime
   must still be connected before physical independence. Ordinary and terminal cleanup readback retain their Wgpu path. They cannot silently join the target before all
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


## Original selective I/O owners and retained replay pins (2026-10-16)

Source `7477b6662aa507b22bdfc5cb8fafa5fd6985dadf` replaces the all-owner native reuse API
with `recycle_native_selected(nonce, certify)`. The original mutable I/O checkpoint
lends an immutable `NativeIoOwners` view directly to THIS stream's Graph callback;
no enqueue, owner mutation/allocation/reordering or other completion source may
interleave before applying its result. The view contains the actual complete
dedicated tracked Wgpu buffers, original request nonce/ledger epoch and intrinsic
INFO pins. Source contents are not read or copied into the inventory.

`Graph::certify_io_reuse` rejects unsubmitted nodes, incomplete/foreign named
recordings, partial/missing/foreign-family original flights before crediting any
completion. In the active original draft scope it verifies request/lane identity,
joins actual lane1, readiness/handoff and current INFO COPY fences. It does not
join unrelated target math merely to reuse a completed INFO upload source, nor
credit a math header/visibility epoch from that copy completion. Outside that
scope it keeps the full actual drain. Existing stream flush still globally joins
before this callback, so this is NOT physical independent execution yet.

The original full owner handles and ordinals populate `strata_io_reuse::Cohort`.
Sorted numeric buffer IDs and a bitmap preserve original slot order, intrinsic
pins and exact request/epoch/count. Incomplete, duplicate, zero, oversized,
repeated or mismatched inventories fail. These are bounded metadata, NOT fence,
numerical, target-ready or erasure authority. Only the owning Graph's genuine
completion/exclusion callback can authorize the unsafe I/O application boundary.
No debug/serialization/content persistence or new GPU root is added.

Every original retained NAMED recording contributes its actual recorded access
buffer IDs, including joined recordings needed for FUTURE direct replay. Every
still-live anonymous recording contributes too. `Plan::buffer_ids` derives this
from its existing canonical descriptor/physical/indirect/copy intervals without
new declarations/allocation. For these dedicated I/O buffers, host reuse writes
the whole API owner: ANY recorded read/write subview conflicts. Main-arena subview
owners are not substituted into this inventory. Anonymous completed cache entries
remain ordinary per-invocation candidates rather than permanent source pins.

Only admitted slots can gain NativeReady or enter the intrusive free index. An
excluded unused native grant is revoked to Idle; an unexpected mapped/staged/
partial Upload on an excluded owner rejects. Full-cohort state/bounds preflight
still precedes any publication. Intrinsic INFO pins remain authoritative even if
a cohort were to select their slot. Original owners, captures, exports, MAIN debt,
full terminal erasure and actual copy/host-write rules remain retained. No owning
storage or failed context is released/resurrected by a bitmap.

The original4MiB I/O metadata ceiling and actual tracked-owner size >=32 bytes
bound this inventory to131072 owners. Actual ID/bitmap capacities are checked
under a new4MiB HOST allowance (not eagerly allocated). `NATIVE_QUEUE_HOST_BYTES`
now includes it:130MiB ->134MiB. This is host bookkeeping, not new VRAM/model data.
The CPU envelope test uses the actual Wgpu/Tracked types and original ceiling.

CPU receipts under `/media/jmicko/faststore/kuno-development`:

- `strata-selective-io-final-test-build-20261016.log`: locked selected release
  test build passes. Earlier unsuffixed build passed but exposed a private-
  interface warning; the Graph method now has matching compute-only visibility.
- `strata-selective-io-cpu-20261016.log`:281 cases across23 nonempty CPU suites
  pass (previous276 plus3 original cohort cases and2 original I/O owner/index/
  envelope cases). Coverage includes sorted original ordinals, intrinsic/live/
  future-read exclusions from actual Plan intervals, empty/maximal/partial/
  duplicate inventories, foreign/stale request/epoch/count, native-grant revocation,
  excluded-slot index denial and unchanged INFO pins. Existing draft scope cases
  additionally check original active request lifetime. These are host logic and
  source inventory tests; no actual Vulkan queue/map/fence is simulated/proved.
- `strata-selective-io-{core,off}-clippy-20261016.log`: strict locked selected/off
  release Clippy pass. Changed Rust formatting and `git diff --check` pass.
  Only existing duplicate-main-bin Cargo manifest warning remains. The I/O CPU
  harness still uses disabled activation/generic lab cfg as described above.

No shader/binding layout, immutable ordinary release, GPU/model/service/protected
inference/driver/API/UI/install/release change. Frozen `0bf0b5f` is stale; build
current source before protected execution. Sudo refresh succeeded in Demons.
Required Vashti changes:none. Full rewrite/physical/numerical/recovery/speed gates
remain open; goal stays active for CPU-only work.

### Next connected graph admission and lifetime block

Implement these together; keep actual final/context/failure drains. No timing
trials or speed-based component rejection:

1. Separate `queued nodes` from `any pending flight`. Stream dispatch must not
   execute a new MTP graph merely because target lane0 is still in flight.
2. Active draft ENTRY/queue selection joins the DESIRED lane1, not the still-
   selected main lane0. Its actual precommit readiness signal/wait already exists;
   do not enqueue another main signal behind commit. Keep unknown old handoffs.
3. Push/copy use actual incoming access ranges and original conflicting flights.
   Execute/replay use current-lane completion plus actual plan conflicts. Direct
   replay needs the retained original full access inventory, not only rebuilt
   nodes. Keep exact original fence/recording identity. Joining sampled work may
   remove a recording: re-find candidates by stable ID instead of stale Vec index.
4. Native-only scoped flush/readback/named first/continuation reuse must keep the
   scoped boundary. ACTUAL dirty Wgpu primary initialization/writes require their
   genuine global completion and cross-queue visibility; preserve cold behavior.
   An INFO-only copy does not manufacture a math visibility/header receipt. Keep
   unconsumed INFO visibility until its real header or final owning handoff.
5. Preserve allocator owners while peer work is pending. Inspect ALL
   `memory_cleanup(false)`/`release_uniforms` sites in flush, direct replay and
   native readback; do not drop temporary managed bindings merely because MTP's
   peer fence completed. Defer them through the original pending context/scope,
   then reclaim after actual joint scope-end/final drains. Retained WgpuResource
   alone is not a managed-allocation binding (the R scope already retains both).
6. Keep original target exported acknowledgement, pending MAIN debt, source R
   loan/generation, numerical/family/PLE/I/O proof and state publication distinct.
   Scope-end/explicit owning sync and terminal cleanup must still join all known
   work, including retired contexts. Ordinary non-scoped behavior stays intact.

These are remaining implementation dependencies, not permission to globally
remove waits. Metadata lookup, memory observation and positive-confidence per-
cell checkpoints must be included in the completed scoped call-path audit.


## Scoped native admission and original allocation lifetimes (2026-10-16)

Source `f9d1e06e9791e38cbd29986925a677daeb2f06ed` implements the preceding connected
admission/lifetime plan. This is source/CPU acceptance only. No speed trial or
speed-based component rejection was used.

`Graph::queued()` separates real unsubmitted nodes from a live peer flight.
Stream dispatch now uses queued work, not `pending()`; the latter remains for
full owning preparation/cleanup checks. Active scope entry/select waits on the
DESIRED lane1 and keeps the actual precommit R readiness dependency. It does not
queue a new MAIN signal behind commit. Ordinary selection remains globally joined.

Push/copy admission validates actual node extents/protected R accesses. Both real
flights are preflighted before any completion credit. Only the selected lane and
actual recorded conflicts join. Real semaphore handoffs establish visibility
when the completed source queue differs from the destination. INFO copies use
their actual full source/target ranges and transfer destination. Direct replay
uses the original retained canonical access inventory (not rebuilt nodes). It
re-finds the stable recording ID after joins can remove profiled Vec entries.
Every actual submission rechecks its full original plan and exact tail identity.

`complete_scoped_submission` applies only inside the original active lane1 scope:
it joins that real lane, existing handoff/readiness and source-copy fences without
joining unrelated MAIN math. Warm native-only stream flush, native packet copies,
first/continuation replay and selective I/O reuse reach this boundary. No target
export acknowledgement, family acceptance, state publication or debt refund is
manufactured. Exact original commit ack and pending MAIN guard remain unchanged.

Actual dirty Wgpu initialization/queue writes retain full genuine drains and MAIN
visibility; fresh roots are still initialized. Ordinary explicit sync and terminal
erasure retain full actual drains, even if an active scope is retired. Scope end
still joins all actual known work, restores MAIN visibility and ends the original
R loan before releasing managed pins. No final/failure wait is globally removed.

An INFO-copy fence can complete before a math header consumes its timeline.
`Uploads::wait/needs_visibility` now depend on the original latest/consumed epoch,
not whether its copy pool is still pending. Copy completion may release original
copy-source handles but does not clear the owed wait. A real math header fence or
fully completed copy-to-selected-to-MAIN handoff credits its exact epoch. Newer
copies cannot gain visibility credit from an older consumer; future timeline
waits remain valid after copy-pool reuse. This is separate from I/O reuse credit.

`strata_allocator_scope::Pins` retains ORIGINAL ManagedMemoryBinding descriptors,
keyed/deduplicated by their stable allocation ID. Wgpu buffers alone would not
prevent the memory pool from reusing an unowned suballocation. The stream begins
pins before fallible preparation/native readiness; resolved and newly reserved
main allocations pin while that scope is owned. Actual uniforms remain retained.
Central memory cleanup/uniform release defers while pins are active, including
explicit server cleanup; successful actual joint scope end releases the matching
request and reclaims deferred memory. Foreign/repeated release rejects; unknown/
failed scope completion cannot publish reuse. Pins contain no data, read/write,
fence, numerical or erasure authority and have content-free Debug output.

Checked actual HashMap capacity and conservative bucket/handle overhead bound
pin bookkeeping to4MiB HOST. `NATIVE_QUEUE_HOST_BYTES`134 ->138MiB includes it;
this is neither eagerly allocated memory nor new VRAM/model backing. The existing
original per-owner I/O exclusion/pins, full copy/padding/host-write/map/erasure
rules remain. No allocation/erasure is credited merely by a smaller queue wait.

Receipts under `/media/jmicko/faststore/kuno-development`:

- `strata-scoped-admission-final-test-build-20261016.log`: locked selected release
  test build passes. Earlier unsuffixed build also passed but preceded the final
  submission/primary/visibility edits; use the final receipt.
- `strata-scoped-admission-cpu-20261016.log`:283 passed across24 nonempty CPU
  suites (previous281 plus original-descriptor pin/dedup/foreign-release case and
  copy-completed-but-math-visibility-still-owed case). Pin test uses actual runtime
  handles/bindings and proves their descriptor prevents `is_free`; it is not a
  simulated GPU/fence. Existing I/O wrapper limitations remain (disabled native
  activation/generic lab cfg). No CPU test establishes physical overlap, native
  map/coherence/selected limits, numerical output, erasure/recovery or throughput.
- `strata-scoped-admission-{core,off}-clippy-20261016.log`: strict locked selected/
  off release Clippy passes. Changed Rust rustfmt and `git diff --check` pass.
  Only existing duplicate-main-bin manifest warning remains.

No new shader/binding layout;330 offline shader validations were not repeated.
No new frozen ordinary release artifact; frozen `0bf0b5f` remains stale. No GPU,
model, service, driver, protected inference, install, API/UI, release or deployment
change. Sudo refresh succeeded in the original shared Demons shell. Required
Vashti changes:none. Goal remains active for CPU-only continuation, not complete.

### Next source audit and remaining gates

Audit the entire now-connected scope against pinned Strata's generate11100/draft/
verify/commit and queue ownership paths. Include first/continuation/capture/direct
replay, per-cell positive-confidence observations, metadata/scheduler dispatch,
INFO epochs, host/source lifetimes, pending MAIN debt and terminal failure drains.
Warm native internal flushes are scoped; explicit ordinary sync/cold primary work
still fully join by design. Confirm no hidden ordinary sync or unexpected cold
root is required in steady state before claiming independent execution.

General arbitrary request entry remains public-fixture gated; prompt/context
capacity still differs. Source89da25f below resolves embedding placement in source;
actual physical acceptance and exact dense pricing remain open. Old core
regression failures remain untriaged. Physical queue/event/lifetime, full numerical
and actual recovery acceptance remain open. No speed testing until the COMPLETE
connected source audit passes. Renew GPU authority before protected device work;
current authority remains CPU-only despite sudo availability.


## Shared native embedding in original HOST backing (2026-10-16)

Source `89da25fb6e7c4b0a30a562a188f7d27473502e40` closes the mapped embedding
placement gap in source. Pinned Intel `generate.cpp:2630..2647` loads NativeEmbed
before model serving, and `sycl/src/core/native_head.cpp:168..290` keeps the packed
table in HOST memory with a GPU alias and GPU gather. This applies to both target
and `sycl/src/core/mtp.cpp:818` consumers. No CPU prompt-dependent row read or
whole-model duplicate is substituted.

`Host::reservation/load` now quotes/constructs16 original IQ4_XS embedding pages
beside the expert complement in the SAME immutable public external registry.
Each page uses the existing actual non-DEVICE_LOCAL HOST-visible allocation,
dedicated-memory envelope/heap checks, concurrent queue families, full initialized
padding, coherent/noncoherent cache publication, HAL import and registration.
Read directly from the authenticated original tensor into its mapped backing;
original packed finite-scale/padding validation is preserved before publication.
Every actual page/shard and full logical/backing size remains owned on failure.

Embedding pages use a disjoint internal coordinate tag (49): neither expert
layers0..47 nor optional prompt shadow48 may alias it. Existing expert pointer
planning, exchange and shadow charges remain; only the original embedding gather
consumers use these embedding page handles. HAL-imported Wgpu29.0.4 buffers use
`BufferInitTracker::new(0)` (`device/resource.rs:1309`), so their externally
initialized weights are not Wgpu-clear candidates. Real first-use primary usage
transitions still occur; no cold initialization/visibility boundary is skipped.

Global sources/stages are fully constructed before Weights and shared MTP. The
shared Weights keeps an original embedding reader loan, with graph binding and
exact same-client checks. Both target and MTP retain that same Weights ownership.
HOST close refuses live loans BEFORE any registry detach. Executor close drops
its last shared weight owner (after real owning cleanup and other-reader checks)
before Sources detach; unknown/live readers keep the original backing retained.
The existing exact detach receipt, actual queue join, source destroy and zero/free
path remains mandatory. No Arc counter is a GPU/numerical/erasure receipt.

Accounting separates three real quantities:

| Quantity | Bytes | Owner |
| --- | ---: | --- |
| Embedding API pages |337715200 |HOST quote (each actual allocation envelope also reserved) |
| Shared Q6 head API pages |521474048 |MAIN |
| Target/MTP addressable original pages |859189248 |Both tiers, one copy each |

`public_bytes` now names the shared MAIN quote; `embedding_bytes` is separate
HOST data; `addressable_bytes` preserves the complete borrowed tensor extent.
MTP additional capacity and constructor admission subtract BOTH borrowed tensors,
not merely MAIN head bytes, so moving the embedding does not charge MTP for a
second phantom copy. Addressable public validation/page shapes remain unchanged.
Internal execution-check JSON records all three; public HTTP API is unchanged.

**Explicit remaining substitutions:** the existing16-page descriptor gather and
tiled head kernels are unchanged. Upstream gathers from one full alias; full
work-sharing/numerical substitution review is still required. The static dense
cache allowance stays5GiB rather than recomputing a larger expert prefix from
observed free memory. This source removes the embedding's actual GPU allocation;
it does NOT prove a larger expert cache or throughput gain. Do not run speed A/Bs
or roll back coupled placement due unaccepted partial timings.

Receipts under `/media/jmicko/faststore/kuno-development`:

- `strata-mapped-embedding-final-test-build-20261016.log`: locked selected release
  test build passes. Earlier unsuffixed build passed but preceded final loan/
  reservation tests and same-client checks; use the final receipt.
- `strata-mapped-embedding-cpu-20261016.log`:301 passed/29 nonempty CPU suites.
  Previous283 scheduling/lifetime cases plus18 placement-related cases: shared
  weight full-page/token/payload/backlog4; MTP shared tile/capacity1; original HOST
  registry/embedding loan5; global plan/reservation5; source table/borrow3.
  New loan case uses real Arc lifetime only; other original registry tests use
  synthetic public integers, not fake device/fence success. Complete GPU/HOST/
  addressable totals, full/partial/no resident expert tiers and unchanged shared
  MTP capacities are checked. These tests do not establish real native allocation,
  alias/coherence, queue execution, math, erasure/recovery or speed acceptance.
- `strata-mapped-embedding-{core,off}-clippy-20261016.log`: strict locked selected/
  off Clippy passes. Changed Rust rustfmt and `git diff --check` pass; only existing
  duplicate-main-bin manifest warning remains.

No shader/binding layout changed;330 offline shader cases were not repeated.
No GPU/model/service/driver/protected inference/install/release action. Sudo
refresh succeeded in the shared shell. Required Vashti changes:none. Goal stays
active for CPU-only work; the full rewrite is incomplete.

Normal immutable selected release also builds successfully (no binary execution):
`strata-mapped-embedding-ordinary-20261016.out`, source89da25f,
SHA256 `6016128a10833a9ef2448209e25f43afe2dac4e4cfbb7c895fa7f43f44fb840c`,
artifact directory `strata-mapped-embedding-ordinary-20261016/`. Its source commit,
build features and Cargo logs are retained there. Builder checked clean source
paths and unchanged HEAD throughout compilation. This replaces0bf0b5f as the
current frozen ordinary source artifact; it is NOT physical/model/speed acceptance.

Next CPU work remains the full connected scope/source audit, general request entry,
prompt/context capacities and exact dense/cache pricing. Embedding placement is now
connected but its physical acceptance and tiled-gather substitution review remain.
Keep the existing full numerical/physical/recovery gates and no-speed condition.


## Connected 4096-row prompt capacity (2026-10-16)

Source `bcc30dde7c25e1f24fa7304f0a72b3e834fcebaa` expands the CONNECTED prompt
capacity, not just its count guard. Pinned Intel prefill.cpp::Alloc/carve
(298,1128) sizes all retained residual/GR/PLE and overlaid GDN/QSA/MoE buffers from
T. The old2048-row Rust owners/route packets could not safely admit4096 rows.

- Prompt rows2048 ->4096; same MAX8 decode/verifier geometry stays separate.
- Rust actual packet dimensions and router/expert/reduce WGSL now use ONE compiled
  `groups::shader_prefix`: logical offset32768 ->65536, raw65536 ->131072 words,
  grouped map/probability32768 ->65536 words, status65536 ->131072 words, second
  role plane offset20480 ->40960. Full40960 routes cannot overwrite the raw
  logical-ID plane or grouped header/members. Expert address/remap tables keep
  their own original sizes; those are not prompt packet dimensions.
- GU16 ->32MiB, PART256 ->512MiB, working images64 ->128MiB. Native dequant checks
  use the same actual working dimension. Carved original persistent/PLE/GDN/QSA
  views expand together, preserving phase overlays and disjoint live views.
- Three owned roots512 ->1024MiB each; MAIN quote1.5 ->3GiB. Global source/expert
  planning consumes the increased original quote before constructing cache roots.
  Private MAX_SPAN/pool bins (16 powers,32KiB..1GiB), MAIN max page and native
  buffer/binding inventory checks agree. Insufficient actual limits must reject
  before allocation; no Vulkan/device limit or physical support was inferred.
- Double-buffer PLE source HOST64 ->128MiB; prompt HOST baseline64 ->128MiB plus
  existing queue metadata; prompt runtime64 ->128MiB; group HOST1 ->2MiB and
  runtime8 ->16MiB. Existing QSA INFO geometry follows capacity/32. Same original
  complete owners, padding, read leases, error/erasure lifecycle remain required.
- The actual residual linear grid grows to65535 x3 groups at4096 rows. Its CPU
  coverage case follows requested count and proves every logical word exactly
  once instead of retaining the old2048-row two-Y expectation.

No arithmetic order, target state proof, private-content persistence or public
API change. The old 8-slot raw ring/two dequant slots remain the selected upstream
mode. No individual speed test or speed-based rollback was used.

**Explicit remaining substitutions/limits:** allocating three full1GiB roots is
conservative relative to upstream Alloc::carve and overlays. Exact lifetimes/
pricing, the unchanged5GiB static dense allowance and implications for the real
ranked expert prefix need review; do not call this equal memory efficiency. Whole
root initialization/erase currently requires actual1GiB storage binding support.
Physical B70 acceptance of that inventory and the expanded owners is unproven.
Larger-context cache loans must not be treated as accepted at the new capacity.

Context remains4096 in the core graph contract, target QSA geometry/RoPE/positions/
pooled selection and MTP context/cache bounds. The public fixture's final-token
handling cannot exercise a full4096-row prefix plus generation until the separate
32768-context expansion is connected. Arbitrary requests/general lifecycle are
also incomplete. Do not label whole-engine capacity/numerical acceptance complete.

Receipts under `/media/jmicko/faststore/kuno-development`:

- `strata-prompt4096-final2-test-build-20261016.log`: final locked selected release
  test build passes. Initial unsuffixed build failed (Wgpu29 limits areu64, notu32);
  corrected/final builds passed. A stale MAIN test quote still said1.5GiB, then
  failed before correction; the real admission already priced3GiB. Final2 includes
  that corrected expectation and the expanded grid/inventory tests.
- `strata-prompt4096-final-cpu-20261016.log`:320 passed across33 nonempty suites.
  Previous301 plus16 prompt cases and3 limit/pool/full-grid cases. Full4096 route
  roundtrip/corruption, exact/short/aliased parent views, all original quant pairs,
  raw/dequant slot identities, static payload/rounding, actual full-page admission,
  complete MAX_SPAN grid and too-small device inventories are covered. No device/
  fence/map/physical placement, numerical model or erasure/recovery proof.
- `strata-prompt4096-final-{core,off}-clippy-20261016.log`: strict selected/off
  release Clippy passes. Rustfmt/whitespace pass; existing duplicate-bin warning.
- `strata-prompt4096-ordinary-20261016.out`: normal locked selected release/freeze
  passes at sourcebcc30dd; SHA256
  `d702405db8e7d9d6195af21b115a851194176e12a2bb95c1ef692cfa9560b93d`.
  Artifact directory `strata-prompt4096-ordinary-20261016/` contains source/features/
  Cargo records. Stable clean source/HEAD checked; supersedes frozen89da25f.
- `strata-prompt4096-shaders-20261016.log`:330 offline shaders pass, including
  native cooperative matrix/physical-pointer validation using the matrix-aware
  spirv-tools PATH. Frozen `--validate-port-shaders` returns BEFORE adapter/device
  creation. This is shader validity, not GPU execution or numeric acceptance.
- **Additional failed check:** `strata-prompt4096-runtime-cap-20261016.log`,
  `resident_model::batch::tests::frozen_down_integer_exact_owner_and_independent_gu_down_budget_increments`,
  fails at line7357 BEFORE its runtime-cap assertion: logical_words32768 vs old
  expected16384. Parent source has the same16K assertion; frozen_down_integer's
  unchanged eight-row selector supplies32K in the core. No claim that this extra
  test passed or established the new runtime cap. It belongs in the outstanding
  old core fixture/regression audit. Earlier stopped/remaining exploratory CPU
  logs are not the final320-case receipt.

No GPU/model/service/driver/protected inference/install/release work. Sudo refresh
succeeded in Demons. Required Vashti changes:none. Goal remains active CPU-only.
Next connect32768-context geometry/cache/position/RoPE/pooled/MTP and request entry,
finish source scheduling/lifetime/substitution audit, then full numerical/physical/
actual recovery acceptance. No speed benchmarks until the complete audit passes.

Follow-up source `669437d4ad2047cf4356924188d1ac734c9e4bf0` completes the actual
native dispatch inventory check. The real prompt/whole-owner grids use65535 X
workgroups, so the earlier16384 minimum was insufficient even though exact word
coverage passed. Selected native inventory now requires65535; legacy mode remains
16384. The inventory test supplies each actual bound and rejects exactly one less.
No shader/source arithmetic or buffer binding ABI changed in this follow-up.

Final follow-up receipts: `strata-prompt4096-grid-test-build-20261016.log`,
`strata-prompt4096-grid-cpu-20261016.log` (same320/33 nonempty groups),
`strata-prompt4096-grid-{core,off}-clippy-20261016.log`, Rust formatting/whitespace.
The330 offline shader receipt at bcc30dd covers the identical unchanged shader
sources; it is not a physical dispatch/inventory test. The separate old16K-vs32K
legacy case still remains unresolved; do not imply the entire test suite passes.

Latest ordinary freeze: `strata-prompt4096-grid-ordinary-20261016.out`, source
669437d, SHA256 `b6febebaad8e980ce3dfde92ff0094b5beb4bc53c0d0a1da6d64305667c56b0f`,
artifact directory `strata-prompt4096-grid-ordinary-20261016/`. Locked normal
release builds with stable source/HEAD; supersedes the pre-guard bcc30dd artifact.
No device/model execution or new performance result. CPU-only scope remains.

## Connected 32768-cell context capacity (2026-10-16)

Source `8d301b1c1ec8dea8972dddb05ac62b189bac1ecb`. This supersedes the prior
4096-context limitation in the 4096-row prompt receipt. It is internal selected
Intel-core work, not a public API or device-acceptance claim.

The original bound model/reference geometry remains unchanged. The selected
runtime derives its 32768-cell capacity from the same native retained-control ABI
constant. Actual resident model/GDN/QSA construction, batch/executor/router bounds,
result packets, request coordinates and global prompt controller consume that
runtime geometry. Ordinary/reference context remains4096; default native branch
constructors retain 4096 and the selected MTP caller uses explicit context keys.
Target/MTP branch identity still includes immutable capacity, model and state
roles; positions and request data remain runtime controls. Readable cells end at
32767 and a boundary may equal32768. Smaller-context keys retain their own window
validation; increasing the ABI bound does not admit a larger window into them.

Target QSA owns real 32768-row INT8 caches, positions and 8192 pooled blocks. Its
2048-token budget/2051 selection stride and33 sparse attention chunks are
unchanged. Public RoPE tables cover all 32768 positions and have distinct stable
names; both target and draft quote their actual full table pages. Prompt QSA's
pooled/score/closing views grow within the same three admitted 1GiB roots, with
4096-row chunks and unchanged nonaliasing/erase checks. Prompt traversal preserves
all chunks and exactly one final normal decode input. No optional cache loan was
enabled and no owning bytes were invented.

MTP expands both packed KV roots, dense partial/statistics planes, identity
selection, positions, status and guard/rewind boundaries together. Dense attention
uses 512 chunks of 64 cells. Its full cache sweeps require 67584 groups; Guard/Rewind
alone reshape the true linear sweep into 65535x2 and use both coordinates. Other
matrix/token grids retain their original interpretation and reject oversize
coordinates. Full logical cache coverage, inactive suffix, original padded-owner
erasure and fault checks stay separate. Private MTP reservation rises 64 ->128MiB.

The target constructor still checks every actual request-vector sum. Each QSA
stage now owns 81,494,016 bytes; its one shared MAX8 workspace owns 81,690,624 bytes.
The audited12-stage QSA runtime quote is 18,874,368 bytes. Full model/private,
public-vector, MTP table and I/O quotes expand with the real owners. Target/public
language runtime envelopes rise 256 ->512MiB to cover the combined prompt,
verification, checkpoint and MTP reservations; actual I/O remains individually
tracked and bounded. Suffix history now covers 32768 entries instead of 8192 in the
core, with its full two-times-size index charged inside the existing 4MiB fixture
HOST envelope. The sampler's 4096-token sliding history remains intentionally
separate. The ordinary suffix capacity remains 8192.

Final CPU evidence (all under `/media/jmicko/faststore/kuno-development/`):

- `strata-context32768-test-build-accepted-20261016.log`: locked release test build.
- `strata-context32768-cpu-distinct-20261016.log`:371 distinct passed cases in40
  nonempty suites, no failures. The prior un-deduplicated final log records377
  executions/41 suites (six repeated cases); do not count those as377 unique.
- `strata-context32768-{core,off}-clippy-20261016.log`: both locked release strict
  `-D warnings` configurations pass; Rust formatting/whitespace pass.
- Source/owner checks exercise full context prompt traversal, exact actual target
  request totals, full QSA stage/shared pages/I/O, MTP private/public plans,
  complete packed cache grid coverage, last cells, branch/model roles, narrow-key
  rejection, boundaries and full suffix history. No adapter/model is created.

Earlier exploratory first/checked logs include superseded fixed-size test
expectations; they are not the final receipt. The prior historical old frozen-down
16K-vs-core32K scratch test remains unresolved and is not counted as passing.
Full suite/physical acceptance is still open. Two old language tests were corrected
to test compiled-vs-native geometry and actual selected4/8-row permission bounds.

Final ordinary artifact: `strata-context32768-ordinary-20261016/`, source
`8d301b1c1ec8dea8972dddb05ac62b189bac1ecb`, SHA256
`5b0009325381cfa686ae58e1f095b6ce9446be64c18432bad53ba876d2e7ff7a`.
The locked normal builder verified clean source paths and stable HEAD, froze the
executable read/execute-only, and its hash was independently checked. This
supersedes the earlier669437d artifact for this source path.
`strata-context32768-shaders-20261016.log` ends with330 accepted shaders, including
native physical-pointer and cooperative matrix variants, using the matrix-aware
spirv-tools PATH. Only frozen `--validate-port-shaders` was executed; main returns
from that branch before any adapter/device creation. This is offline validity,
not numerical, physical limit, model execution or throughput acceptance.
No GPU, model inference, service, driver, install or release action occurred.
Arbitrary/private request entry, exact dense/prompt Alloc::carve pricing, original
queue/lifetime/substitution audit, legacy regressions and full numerical/device/
recovery acceptance remain open. No speed trial is permitted until the whole
completion gate passes. Required Vashti changes:none. Goal remains active CPU-only.

## Connected transient native request owner (2026-10-16)

Source `dc7ace7823c860b7237535cfb159f3bca2dab687` connects the selected Intel
constructor, request/controller and cleanup path in `strata_engine.rs`. The
compiled public diagnostic now calls this actual entry before its legacy path;
this is not an unused request helper. The controller's new authoritative module
is `strata_engine::controller` (same `strata_global_controller.rs` source file).
It no longer imports the diagnostic's fixture constants or returns serialized
receipts. Its generation, routing, draft/verification and accepted-prefix
algorithm remain the same connected implementation.

`Request::new` takes an owned `Zeroizing<Vec<u32>>` and request-local sampler,
speculation and confidence controls. Tokens, capacities, EOS, sampling values,
confidence and total prompt/output extent are bounded before construction. Native
output budgets can use the remaining 32768-cell context instead of inheriting the
public fixture's512-output cap. The same expanded budget reaches the original
request-I/O geometry/authority. Full prompt/output/suffix HOST accounting stays
inside4MiB. All sampling values and token allocations have wiping owners; neither
request nor completion has content serialization or Debug output.

`execute` takes already authenticated/bound mapped sources and the exact admitted
client/device/queue/limits. Its unsafe contract explicitly retains the original
full-artifact authentication, same target-file identity, protected swap/core-dump
policy, redacted panic hook and exclusive MAIN/HOST/driver obligations. These are
caller obligations, not proofs manufactured by request metadata. The public
caller retains its existing real Intel memory preflight before entering this
function. The engine preserves the original constructor reservations for source,
router and integer planes; global activation still rejects legacy bank execution.
The caller-selected immutable backlog policy is configured and checked after
cleanup. No fallback changes the selected native path.

Every constructor phase is retained outside the fallible inference closure. Body
errors/unwinds go through checked draft close, target close/audit/drop, executor
close/drop (last shared-weight reader before source detach), shared-table drop,
actual shared-root physical cleanup and final draft drop. Original MTP profile
close validation, source-change checks and complete target/source zero audits are
still required. Failures use fixed categories with no backend/request payload;
cleanup failure is recovery-required, never healthy completion. Failed runtime
configuration grants no ownership or cleanup authority over a preexisting peer
runtime. Destructor failures do not suppress attempts to close later owners.
The callback's actual successful token count is checked against the controller's
completion count. Completion is returned only after cleanup, with content-free
counts and erasure receipts; tokens leave only through the caller's callback.

The engine writes no files, receipts, stdout previews or request logs. The public
harness separately owns its explicitly synthetic token output, preview and
content-free receipt; the existing protected runner displays the public preview
without persisting it. It retains the512-output fixture cap. This separation does
not turn the benchmark CLI into a private production worker.

CPU evidence under `/media/jmicko/faststore/kuno-development/`:

- `strata-request-owner-build-final-20261016.log`: locked release test build.
- `strata-request-owner-cpu-20261016.log`:374 distinct cases in41 nonempty suites,
  no failures; includes all preceding focused coverage, relocated controller
  coverage, varying-token/full-context request partitions, invalid parameters,
  bounded heap ownership and the checked unwind/error boundary.
- `strata-request-owner-{core,off}-clippy-20261016.log`: both strict locked release
  lint configurations pass; changed Rust formatting/whitespace pass.
- Intermediate first/checked build/test logs are not the final receipt. The old
  extra frozen-down16K-vs-core32K scratch expectation remains unresolved; this is
  not a full legacy regression pass.

Normal immutable build: `strata-request-owner-ordinary-20261016/`, source
`dc7ace7823c860b7237535cfb159f3bca2dab687`, frozen SHA256
`57ee3ae618eddc40960dfb317f4fd6a69b49cb206117d9e49ab6c945609bbd28`.
The locked ordinary builder checked clean source paths/stable HEAD; the frozen
read/execute-only artifact hash was independently verified.
`strata-request-owner-shaders-20261016.log` ends with330 accepted offline shader
variants under the matrix-aware spirv-tools PATH. Only frozen
`--validate-port-shaders` was executed; it returns before adapter/device setup.
This supersedes the8d301b1 artifact for the request-owner entry. No adapter, model,
service, driver or inference was run during these checks. Sudo refresh succeeded.

**Remaining gates:** the internal Rust entry accepts arbitrary tokenized inputs,
but its current executable caller still uses compiled public fixtures. Connect a
protected, non-persisting arbitrary-input caller/tokenization path before claiming
private request support. The entry owns one full request, including model teardown;
verified public-weight/runtime reuse across requests is not established. Do not
claim a hot multi-request engine from this extraction. Constructor failure,
callback failure, numerical/state and actual shared-device recovery still need
physical acceptance; CPU checks do not prove them. Exact dense/prompt Alloc::carve
pricing, final scheduling/lifetime/substitution audit and old regressions remain.
Full rewrite incomplete; no speed trials until the whole gate passes. No installed
Kuno/API/config/connector/UI change, and required Vashti changes:none.

## Connected prompt phase overlay and 256-query selection (2026-10-16)

Runtime source `12a60bf0615e303409467bccffc7723ddece1395`. The selected Intel prompt path now carves mutually
exclusive PLE/GDN/QSA and MoE views in one original phase root, following pinned
`sycl/src/prefill/prefill.cpp:803-826,1128-1224`. Persistent GR/residual/carry
stay in the first root; the actual eight-slot asynchronous raw DMA ring stays in
an independent root. Two dequantized images remain in separate working space,
with real dequant release before raw-slot reuse. No live phase is allowed to
alias these independent lifetimes. Pending family acceptance retains original
state/fault owners, not overwritten phase temporaries. The original native
interval planner and compute barriers see the shared physical parent/ranges.

The three actual admitted MAIN roots are now `[1GiB,1GiB,32MiB]`, totaling2080MiB,
992MiB below the earlier3GiB. Views use the actual80MiB largest rounded working
image,20MiB GU and400MiB routed partials, with native256-byte gaps/alignment.
All original whole-root ownership, erase, parent-span and attachment checks stay
required. MAIN requests, actual allocator bins, descriptor spans, runtime shader
ABI, source attachment, raw-copy bounds and whole-model pricing change together.
No content is placed in INFO/MAP as a substitute or saved to disk.

Pinned `Prefill::Impl::sel_batch=256` and its2986-3000 score/selection loop now
have a connected consumer. Scores and closing flags each reuse an8MiB view,
rather than a full4096-query plane; complete4096x2051 selections remain live for
subsequent32-query attention tiles. Only2048-byte query rows are rebased. The
selected2051-word stride stays on its actual aligned full owner and uses an
explicit logical destination row in the shader. Decode supplies zero and keeps
its original MAX8 addressing. Extra score/validate/select parameter records are
priced together: QSA prompt INFO6,782,976 ->7,888,896 bytes for all12 layers.

CPU evidence under `/media/jmicko/faststore/kuno-development`:

- `strata-prompt-overlay-build-final-20261016.log`: locked release test build.
- `strata-prompt-overlay-cpu-final-20261016.log`:378 distinct CPU cases in42
  nonempty suites, no failures. Includes actual carved alias/disjoint intervals
  through the original queue planner, raw-copy roles on the actual32MiB parent,
  all full-prompt boundary/tail batches, descriptor offset/parent retention and
  complete INFO record pricing. No test constructs a physical device.
- `strata-prompt-overlay-{core,off}-clippy-20261016.log`: both strict locked
  release lint configurations pass; formatting/whitespace pass.
- Normal immutable release: `strata-prompt-overlay-ordinary-20261016/`, source
  `12a60bf0615e303409467bccffc7723ddece1395`, SHA256
  `b80113f0c7d2f0e4afc1d055200105aa2b81f4457386add05e41dd803b0e3040`.
  Builder source-clean/stable-HEAD checks pass; independent hash and mode0500
  checks pass. Build summary: `strata-prompt-overlay-ordinary-20261016.out`.
- `strata-prompt-overlay-shaders-20261016.log`:330 offline shader variants pass,
  including the actual new selection addressing and changed prompt ABI. Uses
  matrix-aware spirv-tools. Only frozen `--validate-port-shaders` ran; it returns
  before adapter/device setup. This supersedes the dc7ace7 frozen artifact.

This is a connected source/lifetime reduction, not GPU or throughput evidence.
No model/adapter/inference/service/install/API/frontend change occurred; required
Vashti changes:none. Sudo refresh currently fails noninteractively in Demons;
CPU work needs no root. Last physical recovery evidence remains historicalOct15.

Remaining Alloc substitutions: private root bins still round to powers of two,
not native2MiB granular allocations; GR/PLE retain distinct F32/ping-pong scratch
rather than identical native packed/in-place storage. The whole prompt/dense
pricing and final scheduling/substitution audit remain open. Protected arbitrary
input/tokenization, verified hot public-weight/runtime reuse, old regression
reconciliation and full numerical/device/failure/recovery acceptance also remain.
Whole rewrite incomplete; speed gate closed, goal active CPU-only.

## Connected sealed dense inventory and automatic expert capacity (2026-10-16)

Source through `910cb5895a25ff424d2a6f76d8f6e58fa11833da`, comprising `cbf4381`
plus `910cb58` on the complete preceding rewrite; these are not standalone leaves.
This supersedes
older notes that leave the blanket5GiB dense allowance and lazy native LRU in
place. Pinned upstream is still clean `fb58e0d`; `generate.cpp:2638-2792` loads
canonical/native dense owners before `4168-4360` sizes the expert cache from the
remaining memory. Rust retains its existing explicit31GiB shared MAIN ceiling,
separate physical/driver preflight and complete peer/private/runtime reservations.

`executor::native_static` now enumerates784 original target projections,
including the48 original BF16 scalar shared gates as one-row matrices. Every
actual tensor must match the bound graph, original header-derived dtype, exact
input/output/row geometry and complete byte span. Other matrices cannot fall
through an exclusion rule. The saved1224-tensor original-shard receipt proves
coverage of all784; the remaining440 are144 routed tensors, three separately
owned head/embedding/table sources and293 separately prewarmed norm/control/
convolution vectors. No new model file or converted weight copy is created.

Before any GPU allocation, `begin_resident` prices this actual dense inventory
plus the single shared target/MTP head. The native expert request ceiling is now
all512 original experts/layer, clipped by remaining shared MAIN and original
ranked/chunk placement, instead of the retired240-slot uniform fixture. The
ordinary feature-off bank policy/configuration remains unchanged. This does not
bypass actual source binding, RAM reservation or original physical root admission.

At256-byte alignment the public inventory prices3,644,840,960 dense bytes plus
521,474,048 shared-head bytes =4,166,315,008 bytes (3.880183GiB), replacing the
5,368,709,120-byte allowance. Other alignments are computed from the actual client;
4096-byte cases are also checked. This releases1,202,394,112 bytes of capacity
allowance, not a measured physical-memory or performance saving. Public vectors,
private caches, source uploads, driver margin and MTP remain separately priced.

`prepare_global_sources_before_prefill` now preloads all original dense roots
through the actual admitted executor before attaching the expert source owner.
Every upload charges its actual aligned immutable root and full source page,
joins the same original client and obtains the existing real admission checkpoint
before the next upload. The peak source remains64MiB, already within the separately
reserved shared public-source allowance. Exact padded owners retain their original
logical shader views; actual backend root lengths are checked. Allocation guards
use the original transient retirement flag. Extra bounded plan/iterator metadata
is charged512KiB in the native HOST/component/metadata envelopes.

Only after every actual upload/checkpoint and complete cache inventory succeeds
is the cache sealed. Native execution/global source attachment require that seal.
Cache hits must retain the original logical/owned extent; missing, partial or
foreign requests fail instead of evicting or lazily changing roots held by graphs.
Model close retains the existing reader-before-release/source cleanup order and
clears this public inventory with the executor. This is still one full request
per `strata_engine::execute`; it does not establish cross-request hot reuse.

CPU receipts under `/media/jmicko/faststore/kuno-development`:

- `strata-static-inventory-receipt-build-20261016.log`: final locked release test build.
- `strata-static-inventory-cpu-final-20261016.log`:387 distinct CPU cases in44
  nonempty suites, no failures. Actual header/profile coverage, exact projections,
  original-format root/source pricing and all preceding scoped owner/queue checks.
- `strata-static-inventory-geometry-20261016.log`: intermediate `cbf4381` public
  profile-derived byte calculation. The final suite checks the same totals against
  original shard headers; no hardware allocation, inference or timing.
- `strata-static-inventory-final-{core,off}-clippy-20261016.log`: both strict
  locked release lint configurations pass. Formatting/whitespace pass.
- The extra initial `strata-static-inventory-profile-20261016.log` found one
  stale test mixing the144-slot reference with compiled240-slot constants.
  Its reference now uses explicit144/72 geometry and independently validates
  the selected layout against original header bytes. All five profile tests
  pass in the final receipt; no production profile/weights were changed.

Shaders/generators/arithmetic are unchanged in this block; the preceding330
shader receipt is not a new runtime/GPU acceptance. No new frozen ordinary
artifact was built; the12a60bf frozen binary is stale for these new constructors.
Freeze a new current-source artifact before any authorized execution check.

No GPU/model/service/install/API/frontend/release work. Required Vashti changes:
none. Last sudo refresh failed; root is not needed for CPU continuation. Historical
Oct15 recovery remains historical. Whole rewrite incomplete, goal active CPU-only,
speed gate closed. Remaining gates include private prompt allocator/packed/in-place
substitutions and whole-device pricing, protected arbitrary-input/tokenization,
verified hot model/request reuse, final scheduling/source audit, unresolved old
regressions and full numerical/device/failure/recovery acceptance. Do not claim
actual new resident counts, a larger observed cache, correctness or throughput.

## Connected retained target/MTP request erasure (2026-10-16)

Source `4ceb86f104114f9b89293a59641cb5e3cd597e24`. The lifetime audit found that
the only existing retained target reset was the construction-fixed13-token
public replay. Its permit and original scope remain unchanged. Native completed
requests now use a separate scope with actual1..32768 context/cursor bounds,
global original-source authority and a closed language I/O scope. A metadata
cursor or a count-only audit cannot authorize this operation.

The shared coordinator prepares real original GDN/QSA/PLE family permits before
mutating state, checks the original fault before erasure, joins the original I/O
and source ownership, then erases/inspects every tracked MAIN/source owner including
padding. Only the completed fault/I/O/final checks allow infallible family cursor
publication. PLE host convolution/history reset to the actual initial sentinel;
the genuine one-input `[-1,token]` predecessor history is accepted without relaxing
the13-token diagnostic. Original buffer IDs, epochs, generations, stamps, policy
choices, model weights and admission charges remain owned; no failed/retired gate
is cleared. Native failure returns to the engine's existing ordered terminal
cleanup rather than independently closing the shared executor under a live MTP.

MTP separately checks its original source/fault, live-output/front state and
shared admission, rewinds through the existing cache consumer, then erases/audits
its entire original private arena. Exact current owner counts, source and real I/O
checks precede a healthy return. Both families issue non-cloneable receipts tied
to their original fault BufferId/model binding; target also retains original
context and closed scope. The engine validates those owners, not a fresh-generation
heuristic. Current generation0/unstarted PLE permission is intentionally unchanged.

The actual `strata_engine::execute` now invokes these consumers after successful
generation and scope exit, before its existing terminal model close. Final close
independently audits the same target and optional draft private owners. Source
close counts remain cumulative across earlier source resets, so the comparison
requires coverage, not equality with the current retained source inventory.
Retained erasure does not release owners or issue allocator/refund/close credit.
Failure never returns a successful generation/cleanup completion.

The32K MTP cache's inspection window exceeds its normal16MiB packet allowance.
`private_arena::OWNED_ZERO_AUDIT_HOST_BYTES` now prices one cleanup map plus its
guarded copy (64MiB for the actual32MiB window) separately in native HOST/component/
metadata admission. Target/MTP scans are sequential; normal MTP read/upload limits
remain unchanged. The real original cache layouts and scan-window bound are
checked together. Reset permit/receipt stack metadata fits the existing8KiB
checkpoint allowance; no new GPU weight, private arena or kernel is added.

CPU evidence under `/media/jmicko/faststore/kuno-development`:

- `strata-native-request-reset-build-receipt-20261016.log`: final locked release
  test build, including the separately priced inspection window.
- `strata-native-request-reset-cpu-final-20261016.log`:402 distinct CPU cases in50
  nonempty suites. Includes original13-only permission/final-fault tests; native
  completed-scope/cursor bounds; short PLE sentinels; original owner metadata
  sizing; actual cache/inspection HOST bounds; no-close-credit/padding/late-failure
  erasure checks; counts-only final-close comparisons and previous connected
  owner/graph/source coverage. No test creates a physical device.
- `strata-native-request-reset-final-{core,off}-clippy-20261016.log`: both strict
  locked release lint configurations pass; changed Rust formatting/whitespace pass.
- Earlier CPU logs are intermediate: one stopped at an empty old `device` test
  namespace, corrected to actual `resident_mtp::device_cache`; another overlapped
  a still-running build and is explicitly retained as `cpu-before-checked-build`.
  Only the final log above was run after the final build's verified completion.

No shader/generator/arithmetic change, new normal freeze or GPU/model execution.
The old12a60bf frozen binary is stale for current source. Services, installed
binaries, model files, API/config/connector/UI and release state are untouched.
Required Vashti changes:none. Goal remains active CPU-only; speed gate closed.
Last sudo validation failed; this work needs no root. Oct15 physical recovery is
historical evidence only.

**Next connected lifetime work (not implemented by this receipt):**

1. Extract a retained loaded-model/session owner from the actual entry. Keep
   original shared source/target/MTP/runtime ownership and terminal close ordering.
   The current wrapper STILL closes everything after one request and therefore
   performs retained erasure plus final close inspection. Do not call this hot reuse.
2. Add a genuine request-capacity/binding lifecycle. Current geometry/context and
   `Prefill -> Active -> Closed`, one-scope permission are single-request. Use
   monotonic request generations and original erasure receipts to rearm; never
   reset ledger/owner epochs, counters or poison flags to manufacture a fresh model.
3. Audit every position0 gate. Current unstarted PLE requires generation0 and
   unselected mode flags; an erased used model is distinct. Preserve selected
   FourGdn/PLE/grouped policies and validate reuse rather than selecting them twice
   or clearing flags to impersonate the constructor. Preserve original graph IDs.
4. Clear request-derived retained HOST/control values before accepting private hot
   reuse, while retaining public weights and compiled program metadata. Current
   proofs cover tracked arenas, known PLE histories and I/O, not a whole-process
   memory-forensics claim. Connect protected arbitrary input/tokenization afterward.
5. Complete remaining private allocator/substitution/whole-device pricing and source
   scheduling audits, old regressions and full numerical/device/failure/recovery
   acceptance with fresh authorization/artifacts. No speed trials before the full
   connected rewrite gate passes.


## Connected retained loaded-session lifecycle (2026-10-16)

Source `1f4b195560e5b8e7a83361f19c844a7104d11cbf` supersedes the first three next-lifetime steps in
`4ceb86f` above. The actual `strata_engine::execute` delegates to the same original
`Session::load`, `run` and `close`. Construction populates the long-lived optional
owners phase by phase; initialization/inference failures and explicit close retain
all-attempt draft-before-target, target-drop-before-executor/source-detach, tables,
shared-root physical audit and final draft-drop ordering. Failed configuration
never grants access to a preexisting runtime. A terminal close is attempted once;
Drop is only a best-effort backstop. The one-shot wrapper explicitly closes even
on a nonmutating run preflight rejection so a cleanup failure cannot be hidden
behind `Rejected`. The session cannot move to another thread because native table
and recording scopes are thread-bound.

The loaded model owns32768 context capacity, separately from each bounded request's
actual prompt/output geometry. Language scopes increase monotonically. The next
request consumes the original non-cloneable target reset receipt and validates
the corresponding original MTP receipt; a stale scope, active I/O, different owner,
capacity mismatch or exhausted scope cannot rearm. A distinct retained-ready
capability checks original fault/graph/context, healthy original admission/owners,
erased PLE sentinel/cursor and preserved generation/policies. It never impersonates
a generation0 constructor or resets epochs, family counters or retirement flags.
Initial source construction remains restricted to the genuinely unused model.

Cold prefill selects FourGdn, grouped PLE and all three grouped-status policies
once. Subsequent prefills validate and reuse those actual selected policies.
The real request controller captures the newly issued language scope and checks
that exact scope at exit instead of assuming1. Successful generation erases and
checks actual target/MTP private owners, drops request-local wiping values, then
publishes Ready with original receipts. A subsequent run uses those loaded owners;
no weights/source copies or new GPU arena are introduced. Terminal close still
independently compares actual erasure receipts, with cumulative source coverage.

This connects reusable ownership in source, **not device-verified hot inference**.
The current public executable still intentionally uses the one-request wrapper.
No arbitrary private input/tokenization caller or installed gateway is selected.
The existing request-end sanitizer drains original graph/I/O owners, destroys
completed recordings, clears parameter templates and binding/header caches, and
wipes actual original GPU I/O. Compiled pipelines and public model weights remain;
this block does not retain whole recording programs across different requests.
A complete retained HOST/control-value audit and physical repeated-request
numerical/failure/recovery checks remain required. Do not infer whole-process
forensic erasure or performance from a CPU lifecycle test.

CPU evidence under `/media/jmicko/faststore/kuno-development`:

- `strata-retained-session-checked-build-20261016.log`: final locked release
  test build, observed terminal exit0 before the following CPU run. Earlier
  `build`/`final-build` logs precede later source edits and are intermediate.
- `strata-retained-session-cpu-final-20261016.log`:404 distinct CPU cases in50
  nonempty suites. Includes repeated varying request/capacity scope transitions,
  stale/duplicate/active/exhausted-scope rejection, matching session erasure state,
  terminal non-reopening and all previous original erasure/owner/source coverage.
  These are CPU state/contract cases; they do not run a repeated physical model.
- `strata-retained-session-{core,off}-clippy-20261016.log`: both strict locked
  release lint configurations pass. Changed Rust formatting and whitespace pass.

No shader/generator/arithmetic change and no
new normal frozen artifact. The old12a60bf artifact remains stale. No GPU/model/
service/install/API/config/connector/UI/release changes; required Vashti changes:
none. Canonical application checkout and inherited untracked files are preserved.
Sudo refresh still reports password required in the existing Demons shell; this
CPU work needs no root. Goal remains active CPU-only; full rewrite incomplete,
speed gate closed and historicalOct15 recovery is not current device evidence.

Next use the retained HOST/control inventory below and connect a protected arbitrary-input
caller/tokenizer, then the remaining packed/in-place scratch/substitution/whole-
device pricing and final pinned-source scheduling review. Preserve the unresolved
legacy regression and numerical/physical/failure/recovery gates. A fresh normal
artifact and renewed GPU authorization are required before device acceptance;
no speed trials before the complete connected rewrite gate passes.


## Retained source controls and full-context sampler (2026-10-16)

Source `a8bbdf0` plus `f1dd77f6a5a5cf98253488edec1fb761c1e9a0a4`. The retained-owner audit found
an actual correctness dependency: full MAIN erasure also clears the GPU's expert
routing map, while paired GPU/HOST weights and their canonical physical placement
remain swapped. Reusing zeros or the initial profile's identity map would select
wrong original experts. Native next-request binding now reconstructs the inverse
from the SAME source owner's committed physical placement. It rejects incomplete
source attachment, pending exchange/loan, foreign graph/client or corrupt duplicate
cells and checks agreement with the existing adaptive cache. It writes the original
full128KiB route owner (including zero padding), independently reads/checks it under
the actual audited I/O boundary, checks the real fault, then rearms the request.
No weights move, and no epoch, adaptive round or ownership counter resets. Usage
scores remained erased at this historical source; source `39c59c1` below supersedes
that reset with bounded model aggregates after full erasure. Physical placement
must remain with the actual weights. Initial loading and ordinary mode are unchanged.

The same full-owner route-map encoder is used by ordinary native adaptive
publication and retained restoration. CPU cases apply twelve real proposal/commit
sequences to the original placement/cache consumers, invert every cell, and check
full packed roots/padding and corrupt/stale mapping rejection. These are structural
CPU consumers, not proof of hardware exchange or numerical outputs. Temporary
inverse49,152B, packet131,072B and original inspected readback fit the existing2MiB
adaptive HOST budget; actual I/O remains under the original runtime admission.
No additional GPU owner/model source or broad whole-device pricing credit is added.

Pinned upstream `sycl/src/program/generate.cpp:6835` fixes only the repetition
penalty ring at4096; request history and positions use the configured context.
The Rust sampler had reused its4096 constant for prefix admission, consumed-token
capacity, speculative handback, controls and accepted-input limits. This silently
prevented the preceding32K request entry from actually reaching sampling. The
selected core now distinguishes32K request context from the unchanged4096 penalty
window. Target/draft penalty histories still take only the last bounded window,
with actual request coordinates/verified-prefix commits checked through32K.
Ordinary sampler bookkeeping keeps its prior4096 bound. BASE/RING offsets, seven full root sizes,
sampler algorithm/shaders, source templates and kernel identities are unchanged.
The existing2MiB HOST quote now explicitly checks full32K consumed-token capacity.

### Retained HOST/control source inventory

| Owner | Request-derived data and actual exit behavior | Intentionally retained |
| --- | --- | --- |
| `strata_engine::Request` and actual controller locals | Prompt IDs, output history, sampler/speculation values, drafts, suffix hashes/history and proposals have wiping owners; they leave scope before Session publishes Ready. | Fixed completion counts only; callback owns delivery. |
| `strata_sampler::request::State` | Thread-local sampling state is taken on normal close; consumed/window/coordinates and Params wipe. Actual seven GPU owners are cleared/inspected. Failure drops state and retires. | No per-request sampling state. |
| `strata_prompt_ple::run` | Prepared packed rows/embeddings and predecessor IDs wipe; bounded producer disconnects and joins before return. Active-scope check now participates in erasure/rearming. | Public GGUF table only. |
| Model/GDN/QSA/PLE and target-row owners | Full original private arenas, caches, padding, PLE token/conv histories clear under the existing actual reset coordinator; output leases and pending stages must be absent. | Original handles, graph identity, generations, selected policies and content-free counters. |
| MTP output/front/private owners | Owned logits/hidden/picks wipe, last forward/front are taken, original private arenas including padding clear and I/O audits run. | Public sidecar/borrowed weights and shape-only replay operation accounting. |
| Native I/O/graph runtime | Existing request-end sanitizer drains original recordings/INFO/flight ownership; drops completed recordings, guarded binding/INFO keys, templates and mutable invocation values; inspects full actual I/O roots. | Compiled public pipelines/IQ/address tables; no tokens, logits or sampler seed in cache keys. |
| Global source/adaptive state | FULL original GPU route/usage owners are erased and inspected. After completed request scope closure, bounded aggregate usage is captured in a wiping model owner. Rearming first validates original zero receipts, then restores original physical routing and aggregate usage with full bit/padding readback. | Loaded-model placement, adaptive rounds and 24,576 aggregate expert counters only; wiped on model close. No token/suffix/seed/KV, serialization, logs or history export. Source `39c59c1` supersedes the former per-request usage reset. The live host job temporarily owns Cache/usage; main publishes only after real worker/fence/visibility joins. Immutable full physical-cell aliases retain no request payload, are built once at load and dropped before normal source teardown; unknown completion retains/quarantines owners. |
| `strata_engine::Session::policy` | Request-local token/suffix/proposal owners still wipe. Only fixed scalar cost/acceptance bins move back to the session after the real controller scope closes; failed requests terminally close the session. | Bounded model-local learned policy, wiped before fallible peer cleanup on close/unload. No request tokens, seed, identifiers, history or export. |

Native reset/rearm now also rejects live sampler, prompt scratch or PLE source
scopes rather than merely assuming their HOST/control guards have closed. This
source inventory is not a whole-process memory-forensics or driver/cache erasure
claim. Physical live request/reuse/failure/recovery acceptance remains required.
The protected arbitrary-input/tokenizer caller is still the next connected task;
current executable diagnostics remain compiled-public and one-request.

CPU evidence under `/media/jmicko/faststore/kuno-development`:

- `strata-retained-controls-checked-build-20261016.log` and
  `strata-retained-controls-cpu-final-20261016.log`: source `a8bbdf0` builds and409
  distinct cases/51 nonempty suites pass. The preceding `controls-build` attempt
  found a sibling visibility error for the new PLE scope query; fixed to crate
  visibility before this checked build/run.
- `strata-retained-controls-sampler-checked-build-20261016.log`: final locked
  selected release test build. `strata-retained-controls-sampler-cpu-final-20261016.log`:
  419 distinct cases/52 nonempty suites pass. This run follows terminal build completion. Includes
  actual placement/cache inverse restoration, long request position/accepted-prefix
  boundaries, unchanged penalty-window target/draft histories and all previous
  original-owner lifecycle coverage. Source/offline sampler Naga/SPIR-V cases use
  no device. New adapter and parent sampler suites are explicit nonempty filters.
- `strata-retained-controls-sampler-{core,off}-final-clippy-20261016.log`: both
  strict locked release lint configurations pass. Intermediate core lint found an
  unnecessary reference in the map equality check; fixed to a slice comparison.
  Earlier context-sampler419-case log predates that lint-only fix. Changed Rust
  formatting and whitespace pass.

No new shader/generator/arithmetic or frozen
normal artifact;12a60bf remains stale. No GPU/model/service/install/API/config/
connector/UI/release work. Required Vashti changes:none. Canonical checkout and
inherited untracked files preserved. Goal remains active CPU-only, full rewrite
incomplete and speed gate closed. Sudo is not needed for this block. Preserve the
remaining packed/in-place/private allocator/whole-device substitutions and final
pinned-source scheduling audit, legacy regression and physical/numerical gates.


## Connected bounded rendered-text input (2026-10-16)

Source `e1458410b20f956586033b359c57186922d7ad8f`. `strata_engine::input::request` reads an already
rendered UTF-8 prompt from a caller-owned `Read` into one wiping allocation, with
an independent1MiB input limit and one overflow sentinel. It preserves control
literals/newlines/whitespace, retries interrupted reads, rejects empty/invalid UTF-8,
oversize and I/O failures with fixed error categories, and catches unwinds under
the enclosing redacted-hook obligation. Reader errors are never formatted/logged.
No command arguments, files, receipts, implicit BOS, normalization or chat-template
rewrite are introduced. The same model's Rust Qwen tokenizer creates wiping IDs
which pass the actual bounded native `Request::new` before inference.

The actual core branch in `daslab_language_probe::run` now supplies its compiled
rendered fixture through this reader/tokenizer consumer and into the original
`strata_engine::execute`. Original exact fixture round-trip/token-count checks
remain; duplicate public fixture IDs drop before the bounded conversion. Native
tokenizer limits now allow1MiB text and up to32767 input tokens, bounded64KiB
pretoken pieces and4M merge steps. Remaining request context/output reservation
is still enforced by Request; overbudget rejects instead of truncation. Native
decode previews have a separate1MiB bound. Ordinary fixture/tokenizer limits stay
unchanged. Original tokenizer model/cache HOST budget remains192MiB.

A separate64MiB transient text/BPE/regex staging allowance is now included in the
original native `CORE_CACHE_HOST`, component and metadata caps. This is separate
from the existing4MiB live generation/suffix request ownership: the input buffer
and tokenizer scratch leave scope before generation. No GPU root or model weight
copy is added. Full physical HOST/cgroup pricing acceptance remains open; this
source quote is not an observed available-memory or allocator refund receipt.

**Boundary still pending:** this is a connected arbitrary rendered-text engine
consumer, not a standalone private process launcher. The executable remains
strictly compiled-public and does not read private stdin or permit overriding a
public fixture/receipt with arbitrary content. `input::request` cannot establish
swap/core protection, model authentication, exclusive GPU admission or a HOST
grant. A separate caller must VERIFY those actual protections before receiving
private bytes, retain the original model tokenizer/source identity, use the
existing retained Session across requests, deliver private output only to the
requesting caller and perform checked terminal cleanup. Do not reuse the public
preview/receipt output branch for private inference. Raw-token Request remains
available for an already-tokenized protected caller.

CPU evidence under `/media/jmicko/faststore/kuno-development`:

- `strata-rendered-input-checked-build-20261016.log`: final locked selected
  release test build, observed terminal exit0 before the CPU run.
- `strata-rendered-input-cpu-final-20261016.log`:421 distinct CPU cases in53
  nonempty suites, including the two new bounded-reader cases and all prior
  original route-map/context/sampler/owner/lifecycle coverage.
- `strata-rendered-input-{core,off}-clippy-20261016.log`: both strict locked
  release lint configurations pass. Changed Rust formatting/whitespace pass.

Reader CPU cases cover split multibyte input,
control/newline preservation, interrupted reads, exact limit, one-byte overflow,
empty/invalid UTF-8 and fixed I/O errors. They do not initialize a GPU, tokenize
the original model or establish private live inference. Whole-model/long/private/
repeated-request numerical/physical/recovery acceptance remains required. No new
shader/generator/arithmetic or frozen normal artifact. Old12a60bf stays stale.
No GPU/model/service/install/release/API/config/connector/UI action; required
Vashti changes:none. Canonical checkout and inherited untracked files preserved.
Goal active CPU-only, full rewrite incomplete, speed gate closed. Root is not
needed for these source/CPU checks. Next connect the independently protected caller
and continue the remaining full pinned-source/pricing/substitution/device gates.


## Direct-source memory owners and legacy regression reconciliation (2026-10-16)

Source `636c9f3543bbb68a188856719beb0262a033a1aa`, isolated
`codex/strata-rust-port`. Pinned upstream remains
`fb58e0dbc8399662c0e47c76578c6e878b14f6cf`, selected Intel
`DEVICE_PLAN=1,NO_HOST=1`. This block completes a concrete dependency of the
already-connected direct-source path; it is not a speed-driven alternative.

The actual selected expert GU/DOWN shaders consume the original immutable VRAM
or mapped HOST cell addresses. They do not invoke the optional decode fetch
helpers. Nevertheless, their bindings retained an unused stage-address table and
three 128 MiB private fetch roots; only the first root had a live consumer, the
dense Q8_1 input preparation. Those obsolete owners/dependencies are now removed
from the selected constructor, source attachment and role bindings together.

Dense preparation now has its own original-arena owner: 73728 logical words,
294912 logical bytes, rounded 512 KiB. It covers every supported packed dtype,
1..8 members and all real input widths through 16384. The actual selected dense
scope and consumers bind that owner. The original prompt roots/raw transfer ring,
real source address tables, every indirect weight-read dependency and source
retirement/drain/erasure responsibilities remain. Source attachment verifies the
original Vulkan device identity before retaining its queue. This source check
is not independent physical queue/device attestation.

The real global MAIN request inventory is 22 roots totaling2,185,068,544 bytes;
the complete target private quote is3,660,939,264 bytes. Compared with its previous
inventory, private owners shrink 402,128,896 bytes (383.5MiB). One unused 32 KiB public
address table is removed as well: total fixed-budget reduction402,161,664 bytes.
Existing automatic expert-prefix sizing consumes the actual remaining budget;
no resident expert count, available physical memory, cache-hit rate or inference
speed is claimed from these source quotes.

Optional explicit fetch diagnostics retain their own caller-supplied 128 MiB
stage destinations and real source binding. They have no dummy stage-pointer
table and remain unselected. The internal lab receipt now reports zero selected
`native_host_stage_private_bytes` and the actual 512 KiB
`native_dense_input_private_bytes`; this is not an application API change.

Before this block, the 31-case `resident_model::batch::tests` suite reproduced
nine failures on the prior selected binary. They were stale profile-specific
expectations: original 4K/reference caps versus actual native 32K state, 8-row
integer storage, replay/INFO ownership and extra priced HOST allowances. Tests
now independently pin each compiled profile and continue checking exact owned
bytes, separate costs, caps, overflow and packet boundaries. No production
admission/cap/policy check was loosened. All 31 pass on selected Intel core,
noncore Strata port and the original no-feature reference target. This reconciles
these nine failures, not every historical broad-core test failure.

CPU evidence in `/media/jmicko/faststore/kuno-development`:

- `strata-core-legacy-before-20261016.log`: prior binary 31 cases, 22 pass/nine fail.
- `strata-direct-source-final-build-20261016.log`: final locked selected release
  test build, verified terminal exit0 before the final run.
- `strata-direct-source-cpu-verified-20261016.log`: 452 distinct selected CPU
  cases / 56 nonempty suites. Runner
  `strata-direct-source-cpu-verified-run-20261016.py` checks actual discovered
  cases, nonempty filters and duplicate exclusion. Includes the full 31 legacy
  cases, 14 actual generated direct-source/fetch WGSL schemas and the original
  dense-size/full-root quote checks, plus prior lifecycle/state coverage.
- `strata-direct-source-off-build-20261016.log` and
  `strata-direct-source-off-legacy-cpu-20261016.log`: noncore Strata port locked
  test build and all 31 legacy cases pass.
- `strata-direct-source-default-lab-build-20261016.log` and
  `strata-direct-source-default-legacy-cpu-20261016.log`: original no-feature
  `kuno-text-lab` locked test build and all 31 legacy cases pass. Attempting the
  feature-gated `kuno-strata-rust` name without its required feature was first
  rejected by Cargo; the manifest gate was preserved, not bypassed.
- `strata-direct-source-{core,off}-final-clippy-20261016.log`: both strict locked
  release lint configurations pass. Changed Rust format/whitespace pass.
  The earlier test build caught missing test-local dense constant imports;
  corrected before final selected/off/reference builds. The earlier 452-case
  `cpu-final` run predates the final core-only test-module cfg guards; use the
  `cpu-verified` log for the committed source.

- Normal immutable release: `strata-direct-source-ordinary-20261016/`, source
  `636c9f3543bbb68a188856719beb0262a033a1aa`, SHA256
  `c0032c4ed9cc065712384deeb624055a7ec1be40af952ee3216986eb6db7b662`. Builder
  clean-source/stable-HEAD checks pass; independent
  hash and mode0500 checks pass. Summary:
  `strata-direct-source-ordinary-20261016.out`.
- `strata-direct-source-shaders-20261016.log`: all 330 offline variants pass,
  including 122 physical-pointer checks, direct-source role/fetch binding changes
  and cooperative matrices using matrix-aware spirv-tools. Only this frozen
  artifact's `--validate-port-shaders` ran; the flag returns before vendor/adapter/
  device setup. This artifact supersedes 12a60bf for current source. It does not
  replace numerical/device acceptance.

No adapter/device/model/inference/service/install/release action occurred. No
application API/config/connector/UI change; required Vashti changes:none.
Whole rewrite incomplete and speed gate closed; goal remains active CPU-only.
The user renewed sudo, but actual shared-shell noninteractive validation still
reported password required; no root was needed for this block. Last physical
recovery evidence remains historical Oct15, not current health evidence.

Continue the remaining native packed/in-place scratch/allocator/whole-device
pricing and final pinned-source/scheduling/substitution audit, then the required
numerical/device/failure/recovery acceptance under renewed GPU authorization.
A private arbitrary-input launcher still requires independently verified actual
swap/core/source/exclusive HOST/GPU protections before reading private bytes.
The current executable remains compiled-public/one-request; retained Session and
bounded rendered-text consumers exist in source. Preserve canonical application
work and inherited untracked files. This cleanup does not establish whole-model
correctness, physical hot reuse or complete upstream equivalence.


## Native prompt GR scales and packed-image consumers (2026-10-16)

Source `37dced95f977428b2be4f616a7e04124bca3953c`, on the complete preceding
`codex/strata-rust-port` engine. This ports pinned MIT upstream `fb58e0dbc8399662c0e47c76578c6e878b14f6cf`
`sycl/src/prefill/kernels.dp.cpp::gr_norm_rs_kernel/gr_silu_kernel/gr_mix_r_kernel`
and their default `prefill.cpp` consumers (`STRATA_GR_UNFUSED` absent/false).
Intel `prompt_f16()` is false; these GR inputs are BF16, not FP16. Optional
HIP-only upmix and the separate opt-in BF16x2 mode are outside the selected Intel path.

The actual native full-prompt GR read no longer stores a full F32 normalized
tensor or an F32 activated-low tensor. Its existing persistent arena carves five
views per token: 5120 words of packed BF16 normalized input, 320 F32 down values,
160 words of packed BF16 SiLU, 10240 F32 gates and four F32 row scales. The same
norm kernel retains original pending-residual write and 256-thread/XOR sum order,
writes one scale per stream and uniquely writes complete BF16 pairs. The SiLU
writer uniquely packs complete pairs. Both keep original finite/fault handling;
no read-modify-write race between neighboring 16-bit stores is introduced.

The original down, optional injection and up projections now consume these exact
BF16 images directly through the existing native Rust cooperative-matrix backend.
They do not call generic Pack or allocate/copy a second image. Only the original
registered normalized/low views are admitted as GR images, with their full parent,
offsets and stream identity checked. Output geometry, fixed original weight
extents, fault extent and read/write disjointness remain checked. External
injection is still owned and validated by the authentic small-arena/full-prompt
caller; it is not falsely required to be an internal carved GR result.

Mixing now reads the actual residual plus its matching row scales and gamma,
reconstructs `(R * rs) * gamma` in the original order and retains the original
FMA/sigmoid/sum order. It does NOT mix the rounded BF16 projection input. Actual
residual `first` is used when recomputing. The source binding and real read
dependencies retain residual/scales/gamma through mixing. The norm interface now
has eight storage bindings plus parameters; mix has six plus parameters.

For a standard GR read with injection, three former Pack dispatches are removed
(two when injection is disabled). At 4096 rows, logical carved payload decreases
86,441,984 bytes (82.4375 MiB); the additional aligned view has its own 256-byte gap.
The three owning prompt allocations remain `[1GiB,1GiB,32MiB]`; this block claims
neither a refund of those fixed roots nor a physically larger expert cache. All
carved views, original phase/raw-ring boundaries and full-root erasure remain.
The existing 32 KiB layout HOST allowance and fixed conservative runtime caps are
unchanged; physical pricing/lifetime acceptance remains open.

CPU evidence in `/media/jmicko/faststore/kuno-development`:

- `strata-packed-gr-build-20261016.log`: locked selected release test build,
  terminal exit0 before the CPU run.
- `strata-packed-gr-cpu-20261016.log`: 453 distinct CPU cases / 56 nonempty suites;
  runner `strata-packed-gr-cpu-run-20261016.py`. New actual layout coverage checks
  exact native image/scale extents, disjoint output regions, identity/slice/foreign
  root rejection and row boundaries 1..4096. Existing real phase/queue alias and
  full legacy owner tests still pass. These are structural/CPU checks, not
  numerical GPU execution of the new kernels.
- `strata-packed-gr-{core,off}-clippy-20261016.log`: both strict locked release
  lint configurations pass. Changed Rust formatting and whitespace pass.

- Normal immutable artifact: `strata-packed-gr-ordinary-20261016/`, source
  `37dced95f977428b2be4f616a7e04124bca3953c`, SHA256
  `fc0d1c655302c774427905e36932797c5c45eb3c02b9b2baf79403be8e198d7b`. Builder
  clean-source/stable-HEAD checks pass; independent
  hash and mode0500 verified. Summary: `strata-packed-gr-ordinary-20261016.out`.
- `strata-packed-gr-shaders-20261016.log`: all 330 offline shader variants pass,
  including new norm/low/mix schemas and 122 physical-pointer checks. Uses the
  matrix-aware spirv-tools build. Only the frozen `--validate-port-shaders` flag
  ran; it returns before vendor/adapter/device setup. This supersedes 636c9f3's
  frozen artifact as current source, not its historical evidence.

No GPU/model/inference/service/install/release or API/config/frontend action.
Required Vashti changes:none. Root not needed; last attempted shared-shell sudo
renewal failed. Last physical recovery stays historical Oct15, not current health.
Whole rewrite remains incomplete, goal active CPU-only, speed gate closed.

Remaining coupled differences are explicit: mixed-input BF16/FP16 image reuse in
attention/MoE, native in-place GR residual/PLE consumers, private allocation
granularity and full whole-device pricing/source/scheduling audit. Retained
target/MTP numerical/state/hot-request/failure/recovery acceptance is still
required under renewed GPU authorization. The current executable is compiled-
public/one-request; independently protected private input remains a separate
unclosed boundary. Do not select or drop this source path based on intermediate
speed. Preserve canonical application work and inherited untracked files.


## Connected native mixed-input image reuse (2026-10-16)

Source `40fe97e6989128189c6046b3b73c64c09288c94a` on the complete preceding `codex/strata-rust-port` engine.
Pinned MIT upstream stays `fb58e0dbc8399662c0e47c76578c6e878b14f6cf`. This connects
`gr_mix_r_kernel`'s F32/BF16/FP16 outputs to the actual Intel `prefill.cpp`
consumers: native GDN/QSA projections use `mixed_h`, alpha/beta/indexer/router
and shared scalar projections use `mixed_bf`, and routed experts use
`gather_rows16(m.mixed_h,...)`. Optional `STRATA_PREFILL_BF16X2` remains off as in
the selected default. It is an upstream opt-in numeric variant, not HIP-only.

Actual `strata_prompt::gr_handles` now queues one pair writer for the F32 result
and both packed images. Each invocation computes two independent columns using
the existing per-column residual/scales/gamma/FMA/sigmoid order, then writes two
F32 values and one complete word per image. There are no overlapping 16-bit
writes. Finite/fault checks cover F32 and both converted images; a conversion
overflow cannot silently authorize later products. Residual `first` is preserved.
The images are disjoint original carved 20 MiB views at 4096 rows, in the SAME
persistent root; they remain independent of the phase/dequant/raw-ring scratch.

A scoped row-count marker is invalidated before each actual GR read and published
only after its real mix writer is successfully queued. It recognizes only the
original full-prompt F32 mixed view, its exact offsets/parent/stream, equal written
row count, zero input-first and 2560 stride. It is a queued dependency, not GPU
completion credit. No user-provided ready flag, receipt, configuration or tensor
name publishes it. New small/tail rows cannot read an old uninitialized image
prefix: counts must match. Scope drop removes this metadata; original full-root
request erasure still clears both images and their padding.

Actual `strata_prompt::project` selects BF16 for original BF16 weights and FP16
for supported quantized matrices when the input matches that queued mixed writer.
This connects existing GDN QKV/Z and alpha/beta, QSA Q/K/V and indexer, router,
shared gate/up/scalar callers without replacing their original F32 handle or
changing weight tiling. The existing native Rust matrix backend directly consumes
the selected original image; generic Pack is skipped. Nonmatching F32 sources keep
their existing explicit conversion; this is a distinct source format, not a
speed-driven fallback. Explicit stage-zero prepacked checks stay unchanged.

Actual `strata_prompt::experts::enqueue` resolves the same original FP16 image
before its logical-expert loop. Gather uses native route/token order and copies
those packed words directly into each group's original working image. The existing
F32 diagnostic source remains supported through an internal checked 0/1 source
format in the same shader; selected full-prompt execution uses the emitted image.
Both source bounds, route bounds, output extents and finite/fault checks remain.
The real descriptors retain these images as readers; new writes and prior readers
remain visible to the original byte-range queue planner. GR preflight includes
both images in external/previous-source overlap exclusion.

Both images together consume 40 MiB of previously unused capacity inside existing
roots. Compared with pre-F1 source, net logical scratch reduction is 44,498,944
bytes (42.4375 MiB), before the added alignment gaps. Fixed MAIN roots stay
`[1GiB,1GiB,32MiB]` (2080MiB): no new GPU root, refund, physical resident count or
speed result. Existing 32 KiB layout HOST allowance and conservative runtime caps
stay unchanged; full physical pricing/lifetime acceptance remains open.

CPU evidence under `/media/jmicko/faststore/kuno-development`:

- `strata-mixed-images-build-20261016.log`: locked selected release test build,
  terminal exit0 before the final CPU run.
- `strata-mixed-images-cpu-20261016.log`: 455 distinct cases / 56 nonempty suites,
  runner `strata-mixed-images-cpu-run-20261016.py`. New actual storage tests cover
  unpublished/next-writer invalidation, BF16/FP16 selection, exact count/first/
  stride/source/parent/slice identity, small-tail reuse, disjoint carved extents
  and actual queue conflict classification for image readers/next writers.
  Previous phase/layout/legacy/lifecycle checks remain included. These do not
  execute the new math or image-copy shader on a GPU.
- `strata-mixed-images-{core,off}-clippy-20261016.log`: both strict locked release
  lint configurations pass. Changed Rust formatting/whitespace pass.

- Normal immutable artifact: `strata-mixed-images-ordinary-20261016/`, source
  `40fe97e6989128189c6046b3b73c64c09288c94a`, SHA256
  `9d1bedf93be55ca5447bfa3ca8d40a29736820b7345335d182c0fac571f3a5d8`. Builder
  clean-source/stable-HEAD checks pass; independent hash/mode0500 verified.
  Summary: `strata-mixed-images-ordinary-20261016.out`.
- `strata-mixed-images-shaders-20261016.log`: all 330 offline variants pass,
  including changed mix/Gather interfaces and 122 physical-pointer checks. Uses
  matrix-aware spirv-tools; only frozen `--validate-port-shaders` ran, returning
  before vendor/adapter/device setup. The 37dced9 artifact remains historical,
  not current source acceptance.

No GPU/model/inference/service/install/release action occurred. No application
API/config/connector/UI change; required Vashti changes:none. CPU work needed no
root; last actual shared-shell sudo renewal was rejected. Last physical recovery
remains historical Oct15, not a current health claim. Whole rewrite incomplete,
goal active CPU-only and live speed gate closed.

Next remaining core gaps are native in-place GR residual/PLE consumers, allocator
and whole-device pricing/source/scheduling substitutions. Full native numerical/
state/prompt/context/hot-request/failure/recovery acceptance is still required
under renewed GPU authorization. Private executable input still needs independent
actual protections/admission before reading content; current CLI remains compiled-
public/one-request. Preserve canonical application work and inherited untracked
files. Do not drop a coupled component based on partial performance results.


## Native PLE key normalization and query-region reuse (2026-10-16)

Source `9d7821a9145be34b190493006e4d6b46c5cc8d03` on the complete preceding `codex/strata-rust-port` engine.
Pinned MIT upstream remains `fb58e0dbc8399662c0e47c76578c6e878b14f6cf`:
`sycl/src/kernels/cuda/native_ple_postops.dp.cpp::native_ple_postops_batch`.
That actual batched path normalizes projected key IN PLACE, normalizes the
original hidden residual into `query_norm`, consumes it for the gate, broadcasts
value/gate into a separate unnormalized `gated`, then normalizes gated into the
SAME query region for convolution/history. Hidden and gated must remain live
through convolution; they cannot be normalized destructively.

Actual `resident_ple_prompt::enqueue_prompt_full` now selects a distinct
`PromptNormInPlace` for key and uses one shared query/conv-normalization view.
The in-place kernel is generated from the original weighted-RMS implementation,
retaining 1024-thread/XOR reduction, gamma row order and finite/status checks.
It has ONE writable values binding plus gamma/status/parameters, not duplicate
read-only/writable bindings to the same buffer. Actual launch rejects mismatched
input/output offsets and non-native strides before enqueue. Reduction finishes
its full input-read phase before per-thread disjoint element writes.

Actual carved PLE inventory shrinks 12 -> 10 unique views: embedding 2560, key 10240,
value 2560, shared query/conv 10240, gate 4, unnormalized gated 10240 and four separate
4-word status views, all per token. The current original key projection owner is
the gate's normalized-key reader. Gate queues its last query reader before conv
normalization overwrites that same exact view. Convolution, NEXT GPU history and
explicit diagnostic tail copy all read the reused conv-normalization view.
Retained history publication, root erasure and original failure/poison ownership
remain unchanged. No duplicate alias entries are added to the owner table.

The other actual consumer, `resident_mtp_prompt::queue_prompt_device`, is updated
in the SAME block. It still borrows a full four-stream hidden plane after target
completion, and now borrows idle PLE views 8/9 for disjoint token/status staging.
It no longer indexes the removed view 10. Test coverage verifies full 4096-row
teacher extents and disjoint original view identities. Target carry ownership and
original MTP/target synchronization are unchanged.

Logical PLE phase payload shrinks 335,544,320 bytes (320 MiB), plus two fewer
alignment gaps. Fixed owning prompt roots remain `[1GiB,1GiB,32MiB]` (2080MiB).
There is no root refund, observed available-memory, residency or speed claim.
Existing layout HOST/runtime envelopes stay conservative and unchanged. Native
GR/residual and final PLE residual output are still separate/ping-pong in Rust;
this block does not claim their full in-place port is complete.

CPU evidence in `/media/jmicko/faststore/kuno-development`:

- `strata-ple-reuse-checked-build-20261016.log`: final locked selected release
  test build, terminal exit0 before the CPU run. The first `build` log reports a
  missing test closure index type; corrected to usize before this checked build.
- `strata-ple-reuse-cpu-20261016.log`: 457 distinct cases / 57 nonempty suites;
  runner `strata-ple-reuse-cpu-run-20261016.py` explicitly includes the new
  `strata_ple::inplace_prompt_tests` module. The earlier 456-case pass omitted that
  module and is superseded. Actual shader schema/Naga validation verifies the
  single read/write values binding and original 1024-thread geometry; layout tests
  verify ten unique regions, gate-read/conv-write queue conflicts, independently
  live gated values, and original MTP teacher bounds. Previous focused coverage
  stays included. This is not numerical/device execution of in-place writes.
- `strata-ple-reuse-{core,off}-clippy-20261016.log`: both strict locked release
  lint configurations pass. Changed Rust format/whitespace pass.

- Normal immutable artifact: `strata-ple-reuse-ordinary-20261016/`, source
  `9d7821a9145be34b190493006e4d6b46c5cc8d03`, SHA256
  `dc1628984b186162f56fe926e2d93e2b2f8536c2059acfe447c67d8abbe9d4b4`. Builder
  clean-source/stable-HEAD checks pass; independent hash/mode0500 verified.
  Summary: `strata-ple-reuse-ordinary-20261016.out`.
- `strata-ple-reuse-shaders-20261016.log`: all 331 offline variants pass,
  including the new in-place norm interface and 122 physical-pointer checks.
  Matrix-aware spirv-tools was used. Only frozen `--validate-port-shaders` ran;
  it returns before vendor/adapter/device setup. The 40fe97e artifact remains
  historical, not current source acceptance.


**Historical source gap, superseded by the fixed-subgroup implementation below:** pinned
PLE norm/gate require `reqd_sub_group_size(32)`. Their WGSL checks 32 but the native
`vendor/cubecl-wgpu/src/compute/strata_graph.rs` registration currently sets a
required 32 only for physical-expert/packed-dense sources (cooperative SPIR-V has
its own 16). For a plain PLE program `required_subgroup` is None, and pipeline
creation sets ALLOW_VARYING_SUBGROUP_SIZE. This does not prove a crash or wrong
size on this B70, but does NOT reproduce the explicit upstream requirement.
Audit all fixed 32 consumers and make both ordinary/native pipeline selection
preserve their exact requirement, keeping cooperative 16 separate. Do not infer
hardware correctness from a source `sw==32` guard or offline shader success.

No GPU/model/inference/service/install/release/API/config/frontend action.
Required Vashti changes:none. Root not needed; last actual sudo renewal failed.
Last physical recovery stays historical Oct15. Goal active CPU-only, full rewrite
incomplete and speed gate closed. Next fixed subgroup fidelity, remaining GR/PLE
residual in-place consumers, allocator/whole-device/source/scheduling substitutions
and numerical/state/hot-request/failure/recovery acceptance remain required.
Protected private executable input remains an independent unclosed boundary;
current CLI is compiled-public/one-request. Preserve canonical work/untracked files.


## Original device subgroup policy and shared pipeline construction (2026-10-16)

Source `5d668c3cdbb669eb5d72bfa93fcccfbb3abe15ad`, isolated `codex/strata-rust-port`. This closes the preceding
source-level PLE subgroup gap; numerical/physical validation remains unproven.
Pinned Strata stays `fb58e0dbc8399662c0e47c76578c6e878b14f6cf`, whose Intel PLE/
GR/GDN/QSA/router/MTP/sampling reductions use32-lane subgroups. Existing Rust
cooperative-matrix products require16 and remain separate.

Actual selected `request_strata_device` now verifies original Vulkan physical
features/properties before creation: API1.1 Properties2, size-control support,
range covering16 and32, required-size compute stage support, and at least32
compute subgroups per workgroup (1024-thread norm /32). Existing subgroup/device
features are enabled by the original constructor. Immediately after that SAME
admitted device is imported into Wgpu, and before any shader/pipeline exists,
its actual HAL owner selects the immutable fixed32 WGSL policy. No replacement
client/device, renderer shadow copy, debug label, shader-name heuristic or
external configuration grants this selection.

The HAL shader factory copies the original device's policy into each WGSL module,
including its delayed Naga/override branch. Standard devices start unset and
retain ordinary subgroup selection. This isolated Intel core conservatively
requires32 for all its non-matrix WGSL programs, covering fixed-width reductions
and width-independent helper shaders together. Existing SPIR-V cooperative
modules retain explicit16; generic non-cooperative passthrough SPIR-V stays
unset. No existing matrix module is overwritten by the WGSL policy.

Both actual ordinary and retained Vulkan pipeline builders now call the SAME
`kuno_subgroup_stage_options` for flags, exact required size and owned stable
pNext storage. Explicit32/16 clears ALLOW_VARYING_SUBGROUP_SIZE. Default/unmarked
modules preserve varying-size behavior only when supported. No REQUIRE_FULL_
SUBGROUPS flag is introduced, so small helper workgroups preserve their original
geometry; only the size-control/maximum-workgroup requirements apply. See the
primary [Vulkan stage contract](https://docs.vulkan.org/refpages/latest/refpages/source/VkPipelineShaderStageCreateInfo.html)
(VUIDs02754..02757) and [required-size structure](https://docs.vulkan.org/refpages/latest/refpages/source/VkPipelineShaderStageRequiredSubgroupSizeCreateInfo.html).
These rules support the implementation; they are not live device evidence.

Retained WGSL registration receives the original Wgpu device from the actual
module creator, reads that actual HAL factory policy and rejects an unconfigured
device instead of inventing a required size from source strings. The immutable
module snapshot drives ordinary compilation; matching original creator/policy
drives retained execution. Retained cooperative SPIR-V continues using the actual
full-module geometry classifier for16. Pipeline pNext owners live through the
synchronous Vulkan creation call in both paths; no temporary pointer escapes.
No shader arithmetic, buffer/root, weight placement, row capacity, GPU allocation
or model count changes. Small content-free compiler metadata remains inside the
existing conservative HOST/runtime quotes; physical pricing acceptance is open.

CPU evidence under `/media/jmicko/faststore/kuno-development`:

- `strata-subgroup-policy-checked-build-20261016.log`: final locked selected
  release test build, terminal exit0 before final CPU run. Earlier owner-level
  attempts failed because Wgpu29 exposes Device.as_hal but not ShaderModule.as_hal;
  replaced with original-device construction policy. Earlier device/final-build
  logs precede the final shared stage helper; use this policy checked build.
- `strata-subgroup-owner-cpu-20261016.log`:463 distinct CPU cases /58 nonempty
  suites; runner `strata-subgroup-owner-cpu-run-20261016.py` explicitly adds the
  actual `strata_matrix::tests` module. The six matrix tests cover original
  workgroup layout/classification plus new actual policy/metadata selectors and
  the SAME helper used by both pipeline builders: default/32/16, idempotent32,
  independent owners, ordinary/delayed modules, varying flag behavior, required
  structure type/size and null pNext. Prior457 focused cases remain included.
  Null-handle metadata fixtures create no driver object, call no unsafe device
  attachment and grant no physical ownership/execution proof.
- `strata-subgroup-owner-{core,off}-clippy-20261016.log`: both strict locked
  release lint configurations pass. Changed Rust formatting (skip_children for
  vendor modules) and whitespace pass. Incidental recursive formatting of other
  HAL files was removed; they are not part of this change.

Normal source-clean/stable build `strata-subgroup-owner-ordinary-20261016/`
completed at this source commit; receipt
`strata-subgroup-owner-ordinary-20261016.out` records SHA256
`0810aa82d7699be35d866ec238010ec2272cb0ea9e27f84d04472e4d4fd27263`. Independent hash and mode0500 checked.
`strata-subgroup-owner-shaders-20261016.log` records terminal exit0,331 accepted
variants and122 physical-pointer checks from this frozen artifact. A preceding
successful invocation printed to terminal; the repeat captured the full durable
CPU-only log. Neither invocation initializes a device.

Source subgroup selection is now connected; driver acceptance, width/correctness
and whole-model execution are still unverified. No GPU/adapter/model/inference/
service/install/release action; CPU checks needed no root. Last actual sudo
renewal failed, physical recovery remains historicalOct15. No application API/
config/connector/UI change; required Vashti changes:none. Goal active CPU-only,
whole rewrite incomplete, live speed gate closed. Next remaining native GR/PLE
residual in-place consumers, allocator/whole-device pricing/source/scheduling
substitutions, protected private input and full numerical/state/context/hot-request/
failure/recovery acceptance remain required. Preserve canonical work/untracked.


## Sole prompt residual and coupled teacher scratch (2026-10-16)

Source `81ff9640afcaf9c520e1915ea668cd49bdef60dc`, isolated `codex/strata-rust-port`.
Pinned `prefill.cpp` owns one prompt `m.R`; `native_ple_postops_batch` updates
that original hidden array. The Rust prompt controller now keeps the same
original `outer[0]` across every attention/FFN half and PLE, instead of swapping
R0/R1. GR norm/pending-write and explicit PLE/final writes select dedicated
in-place kernels from exact original view identity. PLE conv/residual similarly
uses one writable result descriptor. Partially overlapping views are rejected;
disjoint internal callers retain their original distinct shader interfaces.
No duplicate read-only binding aliases the writable R in these kernels.

The original packed GR writer now uses a storage-memory barrier before its
pair-output loop reads residual cells written by other lanes. The existing
workgroup barrier only covered the workgroup sums, not storage-buffer writes;
see [WGSL memory semantics](https://www.w3.org/TR/WGSL/#memory-semantics).
FMA, normalization, projection, mixed-image rounding and causal PLE arithmetic
remain unchanged. The subgroup32/16 owner selection from the preceding block
remains connected. Device numerical/ordering acceptance is still open.

The second residual is actually removed from the carved persistent inventory:
12 outer views replace13, removing160MiB logical payload at4096 rows. The
registered2080MiB roots/admission remain unchanged; no physical VRAM refund or
speed claim. All outer consumers move together, including shared-expert dynamic
indices, mixed-image source identity and diagnostic PLE tail. MTP teaching keeps
target carry `outer[0]` immutable and uses the now-idle original PLE key `phase[1]`
for its residual; `phase[3]` holds normalized hidden, `phase[8/9]` token/status.
The actual all-four-stream4096-row extents are retained, without a new root.

CPU evidence under `/media/jmicko/faststore/kuno-development`:

- `strata-single-residual-accepted-build-20261016.log`: final locked selected
  release test build, terminal exit0 before the final CPU suite. First build
  missed a test-local import; fixed. First CPU run used reserved null buffer ID0
  in a metadata-only planner fixture; corrected to valid independent1/2 IDs.
  `strata-single-residual-first-cpu-20261016.log` preserves that failed run.
- `strata-single-residual-cpu-20261016.log`:467 distinct cases /58 nonempty
  suites. Runner `strata-single-residual-cpu-run-20261016.py` preserves prior463
  coverage, adds original-view/partial-alias selection, original target/teacher
  extents/dependencies, actual single-binding GR schemas and actual single-binding
  PLE conv schema. GR test validates the parsed storage-memory barrier. Existing
  linear dispatch bounds include the new boundary writer. No device initialized.
- `strata-single-residual-{core,off}-clippy-20261016.log`: strict selected and
  feature-off locked release lint pass. Changed Rust format/whitespace pass.

Normal source-clean/stable build `strata-single-residual-ordinary-20261016/`
completed at the source commit above. Receipt
`strata-single-residual-ordinary-20261016.out` records SHA256
`215dc87b810702baa372bb7fd4709a2929695371508c7b0d6ef7728de0616d2c`. Independent hash and mode0500 checked.
`strata-single-residual-shaders-20261016.log`: terminal exit0,334 accepted shader
variants (three new in-place interfaces) and122 physical-pointer checks from this
exact frozen artifact. CPU-only command returns before any adapter/device creation.
No physical/numerical inference acceptance is claimed.

No GPU/adapter/model/inference/service/install/release/API/frontend activity.
Required Vashti changes:none. Whole rewrite incomplete; speed gate remains
closed. Remaining allocator granularity/whole-device pricing, source/scheduling/
substitution audit and full numerical/state/full-context/hot-request/failure/
recovery gates remain. Protected private executable input is independently gated;
current CLI stays compiled-public/one-request. Goal active CPU-only. Canonical
application work and inherited untracked files preserved.


## Exact exclusive pages and shared prompt inventory (2026-10-16)

Source `ba0cc6a815c8f4aa562b7bb386342817e8a0ee30`, isolated `codex/strata-rust-port`. This block replaces the
remaining power-of-two inflation in selected Intel allocation and admission;
source-only payload reductions now change actual requested owning root lengths.
Pinned `prefill.cpp::Alloc::take` accounts256-byte separation and its owned-page
counter rounds individual allocations to2MiB. Rust keeps its three explicit
owning domains (persistent, overlaid phase, asynchronous raw ring), with the same
live intervals and retained queue owners. This is a reviewed backend allocation
substitution, not identical SYCL allocation addresses or driver-page evidence.

The selected original client uses `ExactExclusivePages`: the actual exclusive
pool allocates only the aligned request, ignores rolling-mean enlargement and
reuses only equal-sized complete pages. One pool avoids overflowing CubeCL's
8-bit pool identifiers with512 separate2MiB classes. Its16-bit page-index bound
is checked before storage allocation; a full pool returns an error before a live
descriptor can wrap. Ordinary configurations retain original power-of-two bins.
Selected private owners above2MiB round to2MiB; small status/uniform slabs retain
32KiB/power-of-two shapes. These small slabs differ from upstream per-allocation
2MiB accounting but are complete owned/erased pages, with no borrowed hidden tail.
Persistent exact-source public roots continue their separately aligned pricing.

`strata_prompt_layout.rs` now supplies the actual persistent/PLE/GDN/QSA inventories
to BOTH `Storage::carve` and pre-allocation root pricing, including working images,
all256-byte gaps, full-history tails, routing metadata and the largest live phase.
The final registered root requests are756MiB,642MiB,32MiB (1430MiB), replacing
1GiB,1GiB,32MiB (2080MiB). Raw DMA remains separate. Default4096-row geometry,
32768 context, packed images, grouping, original source identities, subgroup32/16
and all target/MTP arithmetic and state transitions remain unchanged.

The same `BufferLayout` policy now drives private allocation and `MainAdmission`
precharge. Updating only allocation would retain inflated admission and reject
constructor equality checks. All12 target QSA state caches now own50MiB and
pooled keys6MiB (formerly64/8MiB), saving192MiB. QSA verification workspace's
independent32MiB MEAN/POOL buffers stay32MiB; confusing them with the state owners
was caught by the real request-inventory tests and corrected. Total target
private source plan is2,778,038,272B,842MiB below3,660,939,264B:650MiB prompt roots
plus192MiB state. Global-family MAIN is1,503,494,144B. Constructor aggregate
checks remain exact; no admission, poisoning, erasure or recovery guard is relaxed.

MTP32K packed key/value pages become18MiB each and attention12MiB; the complete private
plan (base roles, draft window, native front and control page) is62,586,880B. Its separate128MiB private upper reservation remains
conservative. Only actual output/fault/packet roles are CPU-readable inside the
unchanged16MiB HOST bound; full KV/attention copies remain rejected and native
guards stay on device. The generic Q8 public plan becomes4,203,216,896B (778MiB
below its former native bound), preserving every original tensor and all512
experts. Independent inventory fixtures pin44MiB embedding tiles,120/14MiB expert
shards and807,534,592B dense totals. The selected native runtime Q2 profile is
computed independently and currently prices1,946,574,848B; do not confuse that
runtime plan with the legacy `STRATA_PUBLIC_BYTES` constant for another profile.
The parent's original shared-weight deduction still applies separately.

Remaining power-of-two checks in native mapped-upload/I/O reuse code were
inspected: those consume `transient_io::NativeIoOwners`, whose buffers are created
by the separate direct Wgpu transient factory and retain its original power-of-two
`span` policy. They do not consume these MAIN arena roots. Their strict checks
remain unchanged. The non-native executor cache branch may still explicitly own
power-of-two public pages; the selected sealed native-static inventory uses its
original exact-source branch. No broad validation guard was removed to admit
smaller MAIN owners.

CPU evidence under `/media/jmicko/faststore/kuno-development`:

- `strata-exact-pages-accepted-core-build-20261016.log` and
  `strata-exact-pages-cpu-20261016.log`: final locked selected release test build
  and504 distinct CPU cases /61 nonempty suites, terminal exit0 at settled source.
  `strata-exact-pages-cpu-run-20261016.py` retains previous467 coverage and adds
  actual exact-pool behavior, full private-arena tests and the complete MTP test
  module. New shared-inventory test compares final quotes to actual carved views.
- The actual runtime allocator runs against metadata-only counting CPU storage:
  original12MiB then2MiB allocations, no larger cached-page alias, same-size
  identity reuse,257-byte/512-byte alignment tail and all three exact prompt
  requests. It allocates no GPU or large backing array and is not Vulkan evidence.
- `strata-exact-pages-off-accepted-test-build-20261016.{jsonl,log}` and
  `strata-exact-pages-off-cpu-20261016.log`: ordinary Strata-port test binary,
  54 distinct cases /4 suites covering original arena bins, same real runtime
  pool mechanism, main admission and complete MTP fixtures. Native source does
  not silently change ordinary allocator policy. Exact artifact path is in JSON.
- `strata-exact-pages-{core,off}-final-clippy-20261016.log`: both strict locked
  release lint configurations pass. Changed Rust format/whitespace pass.
- Initial CPU runs exposed retained power-of-two admission, target/MTP32K cache
  and old fixture bounds. Source quotes and constructor equalities were corrected
  together. The QSA state/workspace distinction was corrected without shrinking
  actual verification workspace. Older generic MTP tests now distinguish native
  packed cache roles, legacy flat-grid synthetic caps, actual CPU-readable roles,
  native public tile/shard sizes and the non-native port's32KiB window. Initial CPU snapshots use the first/second/profile/off-first variants of
  `strata-exact-pages-…-cpu-20261016.log` under the same evidence directory. One diagnostic referenced a private constant; removed. First
  lint exposed a now test-only minimum constant; cfg-scoped it, retaining public
  test visibility for existing executor fixtures. Final build/test/lint pass.

Normal source-clean/stable build `strata-exact-pages-ordinary-20261016/`
completed at source `ba0cc6a815c8f4aa562b7bb386342817e8a0ee30`. JSON receipt
`strata-exact-pages-ordinary-20261016.out` records SHA256
`794438e1b0b4f8bbb54752fd46e03f1c0c126b1e1a020e528f14468c6685e88f`. Independent source/hash/mode0500
checks pass. `strata-exact-pages-shaders-20261016.log` records terminal exit0,
334 offline variants and122 physical-pointer checks from that exact frozen
artifact. The offline command returns before adapter/device creation.

No GPU/adapter/model/inference/service/install/release/API/frontend activity.
Required Vashti changes:none. These are CPU/source allocation receipts, not
observed VRAM recovery, new expert residency, numerical/model correctness or
speed evidence. Full rewrite remains incomplete and speed gate closed. Next:
whole-device physical/runtime/HOST pricing acceptance, final pinned-source queue/
scheduling/substitution audit, numerical/state/full-context/hot-request/failure/
recovery gates. Protected private executable input stays independently gated;
current CLI compiled-public/one-request. Goal active CPU-only; root not needed.
Preserve canonical application work and inherited untracked files.

## Original cell transfer copy path (2026-10-16)

Source `8bd6edff399f531565642a66472dfd6afbba0ea1`, isolated `codex/strata-rust-port`. This supersedes the
selected global adaptive exchange's shader mechanism, not the complete source
placement or acceptance contract. Preserve the complete branch; do not select
this leaf separately from its original owners, exact allocator or queue work.

### Connected implementation

- `strata_global_sources.rs::Sources::exchange` remains the actual consumer of
  every `Cache` proposal. It rejects prompt scopes, outstanding prompt loans,
  prior mutations and pending publication. It authenticates all original
  gate/up/DOWN cells against the bound global source plan and original physical
  placement before mutation. The previous client join remains.
- Actual `strata_global_exchange.rs::Host::exchange` now records the same checked
  copy plan tested on CPU: for each original role, HOST to scratch, GPU to the
  SAME HOST location, scratch to GPU. All original bytes survive without
  dequantization/repacking or substituted experts. All three roles complete
  before logical placement/private route-map publication. Bounds, disjoint spans,
  registered HOST capabilities and the original rebase token remain required.
- Scratch is the already-attached complete32MiB prompt raw root. No extra model,
  scratch buffer, duplicate HOST complement, source registration or private root
  is allocated. Actual retained resource metadata keeps that original root alive
  through transfer completion. Whole private-arena reset/close still erases the
  complete root. The bounded copy-plan metadata is below128KiB within the
  existing2MiB adaptive HOST allowance; no residency credit is added or refunded.
- The actual global HOST constructor (`strata_prompt_shadow.rs`) now creates and
  imports STORAGE/COPY_SRC/COPY_DST buffers and matching Vulkan transfer flags.
  The actual public-source registration uses the shared exact usage predicate.
  This adds no mapping permission and grants no write authority by itself:
  original exclusive rebase is still mandatory. Legacy uniform HOST exchange
  and its shader remain separate and unchanged.
- Wgpu records transfer RAW/WAR/WAW ordering on the actual buffers. The actual
  HAL encoder adds TRANSFER/TRANSFER_WRITE to HOST/HOST_READ availability before
  its checked submitted fence; CPU mappings still require their existing cache
  invalidation. Native subsequent consumers retain their existing original
  ALL_COMMANDS memory boundaries and cross-queue timeline dependencies. These
  source contracts do not establish physical driver execution.
- Failed transfer/barrier/fence/error scope cannot publish the proposed map.
  `mutation_pending` keeps target sources unavailable; success still requires
  joined private map/usage writes and `Cache::commit`. No failure is treated as
  permission to infer with partially exchanged weights.

### Source audit: remaining adaptation schedule

Pinned MIT upstream remains `fb58e0dbc8399662c0e47c76578c6e878b14f6cf`.
`sycl/src/program/generate.cpp` serving code around7673/7707 and11080, and its
corresponding generate path around12004/12040/12305, submit adaptive copies on an
independent stream while target commit and MTP drafting proceed. The next
verifier joins actual pending-copy completion before admitting changed residency
(default pending lag1); CPU RAM exchanges and residency publication occur only
when those original copies land. Optional alternate async jobs/lag policies are
not established requirements of the selected single-device configuration.

Current Rust still calls `adapt_global_after_window` AFTER
`verify_native_prefix` returns. That latter method already ran the original
pending target commit and scoped next-MTP observer and joined both. The new copy
mechanism remains on the original Wgpu queue with original whole-client/rebase
joins. **It does not yet overlap adaptation with commit/drafting.** Do not mark
that row complete, call this a complete Strata scheduling port, or open the speed
gate based on CPU tests.

Next connected work must prepare actual usage/proposals at the observed verified
boundary, retain original sources/scratch through an admitted transfer flight,
allow independent commit/draft work, and await the exact original completion and
publish all physical/private maps before the next target reader. Existing public
rebase begins/finishes with whole-client joins; replacing them requires actual
reader/recording lifetime proof, not merely removing waits or creating a second
queue. Only selected expert HOST roots are revoked; the shared embedding root is
independent. Audit the first last-prompt/count1 window's usage/adaptation tick
alongside this change: the current later-window tick alone is not evidence of
matching upstream first-window accounting. That suspected gap is unproven here.

### CPU acceptance and limits

Evidence base `/media/jmicko/faststore/kuno-development`:

- `strata-cell-transfer-checked-build-20261016.log`: locked selected release test
  build completed successfully.
- `strata-cell-transfer-cpu-20261016.log` and its retained
  `strata-cell-transfer-cpu-run-20261016.py`: **508 distinct CPU cases /62 nonempty
  suites, zero failed**. Original parent global suite includes the three new
  copy tests. One duplicate explicit child filter was removed from the runner;
  its rejected duplicate was not a production/test failure or credited coverage.
- Actual copy plan preserves arbitrary exact byte patterns for3/6/9/288 original
  roles with reused scratch. Empty/partial/excess role groups, bad byte geometry
  and scratch overrun are rejected. Actual shared HOST usage and HAL producer
  masks are checked without a device. The fourth new case checks the exact
  descriptor predicate used by actual public-source registration.
- `strata-cell-transfer-{core,off}-clippy-20261016.log`: both strict locked release
  configurations completed successfully. Ordinary54-case receipt is historical
  from the preceding exact-page block; it was not rerun here.
- Changed source format/whitespace checks pass. Two unrelated rustfmt-only panic
  layouts in vendored HAL were restored to preserve the original source lines.

Normal source-clean/stable artifact `strata-cell-transfer-ordinary-20261016/`
(`strata-cell-transfer-ordinary-20261016.out`) has SHA256
`b85b758107e6a34db4e97d053ffa5e98c654fca1842a41bfe22892d9b33abc59`.
Independent source/hash/mode0500 checks pass. Offline
`strata-cell-transfer-shaders-20261016.log` exits0 with334 variants and122
physical-pointer checks. This command returns before adapter/device creation.

No GPU, adapter, model, inference, service, installation, release, API or frontend
action. These checks prove source/CPU transactions and offline validity only,
not physical transfer ordering, source residency/refund, numerical/model
correctness, hot reuse, recovery or performance. Full rewrite remains incomplete;
speed gate closed. No required Vashti changes. Goal active CPU-only. Root not
needed; latest shared-shell noninteractive sudo refresh still reported that a
password is required. Canonical application work and inherited files preserved.

## Adaptive copy flight across commit and drafting (2026-10-16)

Source `53cd14ced31da04c71408e76b266bdec2a2b8c5c`, isolated `codex/strata-rust-port`. This supersedes the preceding
receipt's source-level after-window serialization and suspected missing first
window usage. The full rewrite and physical scheduling acceptance remain open.
Preserve the complete branch, including original/exact owners and queue work.

- Actual `resident_model_batch.rs::run_global` observes target rows, runs the
  original `PrefixPhase::Verified` callback, drops its original host-row lease,
  then starts `Workspace::begin_adaptation_after_verified`. The source cache
  ticks/proposes from actual GPU usage, with original48-layer completed routing
  checks. No target state/cursor acceptance is granted by proposing or copying.
- Nonempty `Sources::submit_exchange` authenticates original three-role cells,
  original attached raw root AND matching actual private-arena allocation. It
  retains the existing original reader/rebase join BEFORE target commit starts.
  `strata_adapt_transfer.rs::Flight` then submits native copies on the actual
  adopted transfer queue, rather than Wgpu's MAIN queue, and returns without
  joining those copies. Commit submission and the scoped next-MTP observer follow
  while this owning flight remains live. All joins are preserved at their real
  boundaries; this is not a removed-wait shortcut.
- Original MAIN signals a real semaphore consumed by the copy queue at TRANSFER.
  The same tested original HOST->scratch, GPU->HOST, scratch->GPU plan is recorded
  with transfer RAW/WAR/WAW barriers and final transfer->HOST_READ availability.
  It still uses the existing32MiB dormant prompt raw root; no extra GPU or model
  root, duplicate RAM complement, new device, queue family or inference library.
- After original commit/draft and all family proof/commit work, actual
  `finish_adaptation` joins the original submitted copy fence, submits a real
  done-semaphore wait to MAIN and joins that original visibility fence. It then
  finishes the original registered rebase, uploads the private route map/decayed
  usage and joins publication/`Cache::commit`. Only then may the next target
  reader enter. Empty proposals still publish decayed usage without a fake flight.
- Actual raw flight owns complete original buffer references, Vk pool/command,
  semaphores/fences and the original Wgpu device through completion, including
  unwind. The same-thread original-arena `strata_external_flight::Block` prevents
  erasure/reset/close while the scratch reader is unjoined. Unknown completion
  retires that arena and leaves its existing quarantine path in force; there is
  no clean-memory/publication credit. Actual completed raw work releases this
  exclusion even if later logical publication fails. Failed/uncertain submission
  never earns completion from its attempt or resubmits an already-used fence.
- The actual first normal final-prompt input now uses a bounded count1
  `prefill_first_native_window` observer returning its real prediction and device
  carry receipt after acceptance. This closes the old usage/tick omission while
  dropping its row lease before adaptation; preceding prompt matrix rows do not
  train the cache. Existing monotonic loaded-cache state/epochs persist. The
  controller no longer ticks again after the completed window. Optional legacy
  diagnostic checkpoint replay no longer trains the selected native cache.

**Explicit substitutions/remaining audit:** Rust computes the small CPU proposal
serially at the verified boundary; upstream launches that selection on a CPU
thread alongside commit/drafting. Actual GPU copy submission is asynchronous.
Vulkan INFO uploads and these adaptive copies use the SAME existing transfer
queue; subsequent draft INFO copies may therefore wait behind adaptive DMA.
The saved Oct14 B70 topology has one queue in each of the three selected families;
this pass does not request nonexistent extra queues, assume independent DMA
engines or claim device overlap. These queue/proposal/first-window delivery
substitutions need final source review and physical acceptance. Current first
count1 returns to the outer controller before its first MTP drafting as before;
later windows use the existing scoped next-draft observer. Original rebase begin/
finish remain full client joins, now outside the intended commit/draft interval.
Model identity, exact quantized math, source complement and private erasure remain
unchanged. CPU tests cannot establish physical queue visibility or throughput.

CPU evidence under `/media/jmicko/faststore/kuno-development`:

- `strata-adapt-flight-final-test-build-20261016.log`: final locked selected
  release test build exits0. `strata-adapt-flight-cpu-20261016.log` and retained
  `strata-adapt-flight-cpu-run-20261016.py`: **514 distinct /63 nonempty suites,
  zero failed**, retaining previous508 coverage. Four additional cases exercise
  the actual transfer-fence state used by real submission/waits: original copy
  completion before visibility, failed/unknown copy attempts, one-shot visibility
  submission, construction without invented completion. Two cases exercise the
  actual arena cleanup exclusion, stale-credit rejection and retirement on lost
  completion. They create no adapter/device.
- `strata-adapt-flight-{core,off}-clippy-20261016.log`: both strict locked release
  configurations exit0. Changed-source format/whitespace checks pass.
- Intermediate `strata-adapt-flight-build-20261016.log` found one remaining legacy
  after-window caller; it was removed with the diagnostic cache-training behavior
  described above. Later `checked`/`scoped` builds are intermediate receipts, not
  the final acceptance/artifact selection.

Normal source-clean/stable artifact `strata-adapt-flight-ordinary-20261016/`
(`strata-adapt-flight-ordinary-20261016.out`) has SHA256
`4fe370cae0c05c30453dd1dd66c1f01933e46017b6ea951016223d09f1deea46`; independent original source/hash/mode0500 checks pass.
The build exits0 without initializing an adapter/device.

No GPU/adapter/model/inference/service/install/release/API/frontend action. No
kernel source changed, so the preceding334 offline variants/122 physical-pointer
checks remain historical; no new shader/device acceptance is implied. No timing
or physical residency/refund claim. Whole rewrite incomplete, speed gate closed.
Next final pinned-source substitution/queue/ownership/pricing audit, then (under
renewed GPU authorization and protected execution) full numerical/state/context/
hot-request/failure/recovery acceptance before speed. The independently protected
private executable caller remains open; current CLI compiled-public/one-request.
Required Vashti changes:none. Goal active CPU-only; root not needed. Latest
shared-shell sudo refresh remains unsuccessful; do not churn on privilege
workarounds. Canonical application work/inherited untracked files preserved.

## Original mapped embedding gather (2026-10-16)

Source `5b520bb50de8bc39387c42e42166ab24437d6447`, isolated `codex/strata-rust-port`. This supersedes the old
per-page embedding dispatch substitution for the selected shared engine. Keep
the complete branch and original model/source/allocator/queue owners together.

Pinned MIT upstream `fb58e0dbc8399662c0e47c76578c6e878b14f6cf`:
`core/verify.cpp` uses `NativeEmbed::gather_dev` then stream broadcast; MTP's
`core/mtp.cpp` uses the same gather before embedding RMS norm. Actual
`core/native_head.cpp::NativeEmbed::gather_dev` calls `iq_embed_rows` on the
original mapped GGUF table. `kernels/cuda/iq_kernels.dp.cpp` lines2530–2549 and
4320–4360 implement IQ4_XS with32 lanes per256-value block: `il=lane/8`,
`ib=lane%8`, one scale per lane and two four-value writes. These are the pinned
Intel/SYCL sources under `sycl/src/`, not an unselected generic packed tensor path.

- Original HOST embedding pages,337,715,200 bytes, remain the same registered
  owner retained for target/MTP. Existing source table0's unused tail now holds
  their16 authenticated addresses/lengths/row counts. Expert entries0–1535 are
  unchanged; embedding entries1536–1551 fit the same32KiB root. No GPU allocation,
  RAM model copy, private owner, model format or residency quote is added.
- `Sources::build_tables` encodes actual same-device full-page addresses already
  authenticated by the original root inventory. `strata_shared_embedding::encode`
  checks complete geometry/ownership/alignment/overflow. This original table is
  retained by `Weights`; original HOST loan/detach rules remain. Prompt table
  rebuilds reconstruct the same embedding tail after their original reader joins.
- `Kernel::EmbedMapped` is the actual shared gather: tokens select the original
  mapped page directly, use the existing native physical-pointer lowering, and
  preserve original IQ4_XS bytes/scales/codebook. One launch has10 groups per
  token (2560/256),32 lanes per group and two float4 stores per lane. Finite
  faults still propagate through the original private fault owner.
- `Weights::gather_queued` checks original client/retirement, private token/output/
  fault extents and non-overlap, and supplies ALL original mapped pages as actual
  native read dependencies plus the retained address table. Native recording/
  replay/handoff retains these actual sources. There is no untracked dummy-page
  access: the first actual page is the logical descriptor, while the selected
  native SPIR-V resolves the table address for every actual requested page.
- Actual target `Weights::gather` still uploads the full wiping token owner and
  broadcasts D into four HC streams after gather. Both MTP decode/front and
  whole-prefix teaching use `gather_queued` when borrowing these shared weights,
  with actual dispatch accounting changed to one gather. Standalone legacy MTP
  without the shared owner keeps its old tiled consumer; it is not the selected
  Session target/shared-Q2-MTP path. No gateway/API/UI change.

Evidence base `/media/jmicko/faststore/kuno-development`:

- `strata-embedding-gather-accepted-test-build-20261016.log`: selected locked
  release test build exits0. `strata-embedding-gather-cpu-20261016.log` and runner
  `strata-embedding-gather-cpu-run-20261016.py`: **522 distinct CPU cases /65
  nonempty suites, zero failed**. This retains previous514, adds four new cases
  and includes four previously unselected front cases. New actual address-encoder
  tests cover every248320 row, original expert/tail preservation, malformed
  extents/alignment/overflow. Actual mapped shader schema/native physical lowering
  passes. Native lane ownership covers all256 values exactly once and matches
  the independent original IQ4_XS decoder for the synthetic packed fixtures.
- Expanding the front suite exposed its old total<8MiB assertion, which predated
  the current32K context attention owner. The test now bounds fixed batch scratch
  separately from its actual context owner. The source allocation quote remains unchanged:
  front8.75MiB, of which the context owner is4MiB, all already charged. Initial
  failure is preserved in `strata-embedding-gather-initial-cpu-20261016.log`.
- Initial `...-build-20261016.log` caught a Rust reference-clone ownership error;
  corrected before accepted build. `...-core-clippy-20261016.log` caught an existing
  allow attribute displaced by method insertion; restored to its actual method.
  Final `strata-embedding-gather-{core,off}-final-clippy-20261016.log` both exit0.
  Changed-source formatting and whitespace checks pass.

Normal source-clean/stable artifact `strata-embedding-gather-ordinary-20261016/`
(`strata-embedding-gather-ordinary-20261016.out`) has SHA256
`6f6e8c34069852c1bd7725bae7a417a54d036c38aa4b70d3e623f63dc4578c77`.
Independent original source/hash/mode0500 checks pass. Offline
`strata-embedding-gather-shaders-20261016.log` exits0 with335 variants and123
physical-pointer checks from that exact frozen executable. No adapter/device is
created by this validator.

Source/CPU/offline evidence does not establish physical mapped reads, numerical
model parity, speed, hot reuse or resource recovery. No GPU/adapter/model/
inference/service/install/release/frontend action. Full rewrite remains incomplete
and the live speed gate stays closed. Next: finish pinned-source head/sampling/
queue/ownership/pricing/substitution audit, then full protected numerical/state/
context/hot-request/failure/recovery acceptance under renewed GPU authorization.
Retain explicit adaptive substitutions from the previous receipt (serial small
CPU proposal, shared INFO/copy queue, first-window outer drafting). Protected
private executable input remains separately gated; current CLI compiled-public/
one-request. Goal active CPU-only; root not needed. Required Vashti changes:none.
Canonical application/inherited files preserved; no privilege workarounds.

## Native aligned wide Q6 heads (2026-10-16)

Source `ffe7293be448bb62ebc48e0ad9c51e11cc79c7af`, isolated `codex/strata-rust-port`. Preserve the whole branch
with original owner/admission/graph dependencies. This supersedes the narrow
32-iqs Q6 lane path as the selected native dense/head implementation; unrelated
legacy Q6 operators and indexed Q8 consumers remain separate.

Pinned MIT Intel/SYCL upstream `fb58e0dbc8399662c0e47c76578c6e878b14f6cf`:
`sycl/src/core/verify.cpp:1540` calls native MMVQ on the whole target head;
`core/native_head.cpp` uses the same original GGUF Q6 weight format. Actual
`kernels/cuda/native_mmvq.dp.cpp:1517..1742,3420..3447` selects wide MMVQ and
aligned A2 loads by default, RPW1/subgroup32, PF0. Packed224-byte weights and
other optional subgroup/RPW/prefetch modes are not selected by this port.

- Actual `strata_dense::Kernel::source` now constructs a read-only vec4<u32>
  weight descriptor for Q6 and uses the rewritten `strata_native_q6.wgsl`. This
  expresses aligned16-byte vector storage reads. Its byte/half helpers retain
  original GGUF coordinates.
  Host projection admission requires the original complete16-aligned padded
  view. Actual target/shared and compact-CJK head pages already meet that contract.
  No weight repacking, new public/private root or new scratch is introduced.
- `load16_a2` reads the original containing aligned16-byte vector and the next
  vector only if the original address is unaligned. Register selects plus an
  optional16-bit shift reconstruct the original QL/QH int4. Q6 blocks stay210
  bytes. Original source payload and initialized padding stay owned throughout.
- One128-thread group covers four rows. Eight lanes share a256-value block and
  four blocks are in flight per32-lane subgroup. Four DP4A operations consume each
  QL/QH vector per half; weights are reused across all actual RHS rows. Signed
  per16-value scales, Q8 half-rounded scales in the existing plane, the wide
  multiply/add ordering and XOR reduction follow the selected source. Private
  Q8-plane transposition remains the existing explicit backend layout substitution.
- Actual `strata_dense::project` now specializes Q6 to exactly the real count1–8,
  instead of rounding counts to1/4/8 and retaining eight accumulator cells.
  Target `enqueue_q6_integer_verify_head` and MTP `cjk::Head::project` consume this
  same operator. Partial output groups preserve bounds: inactive subgroups read
  no weight row and write no output. Indexed English Q8 and other formats retain
  their earlier kernel selection.

**Still incomplete:** `GpuExecutor::enqueue_strata_dense` loops original target
head pages and `resident_mtp/cjk::Head::project` loops compact head pages. Upstream
runs each whole head in one launch. Vulkan needs the original bounded storage
roots, but BDA can address them in one dispatch as embedding now does. Implement
that connected source/address-owner/read-dependency/controller path for BOTH
heads next; do not label page-dispatch behavior complete because the inner kernel
matches or because a later speed number improves. No copying all weights into a
new root, speculative new queue or memory-credit shortcut is authorized here.
Existing private logit/Q8 owners and original head/source lifetimes must remain.

CPU receipts under `/media/jmicko/faststore/kuno-development`:

- `strata-wide-q6-check-20261016.log`: locked selected release check exits0.
  `strata-wide-q6-accepted-test-build-20261016.log`: selected release test build
  exits0. `strata-wide-q6-cpu-20261016.log` and retained
  `strata-wide-q6-cpu-run-20261016.py`: **527 distinct CPU cases /66 nonempty
  suites, zero failed**, retaining previous522. Two new actual wide cases plus
  three newly included dense cases cover every supported RHS and actual CJK tail,
  vec4 storage stride/native SPIR-V generation, and original column/signed-scale
  ownership against independent canonical GGUF decoding for1/2/3/4/5/10/64 blocks.
  They do not execute a GPU dot product or prove matrix/model numerical parity.
- Added dense suite exposed an old fixture treating640 columns as legal for all
  formats.640 remains a supported Q8 producer/shared-expert shape, but cannot form
  Q*_K/IQ4_XS256-value rows. Corrected only the test cases; runtime support and
  scratch quote are unchanged. The first failing run is retained as
  `strata-wide-q6-initial-{cpu,test-build}-20261016.log`.
- `strata-wide-q6-{core,off}-clippy-20261016.log`: both strict locked release
  configurations exit0. Changed-source format/whitespace checks pass. The offline
  source inventory now includes Q6 counts2/3/5/6/7 as well as1/4/8 and the real
  compact-head full/tail row geometries.

Normal source-clean/stable artifact `strata-wide-q6-ordinary-20261016/`
(`strata-wide-q6-ordinary-20261016.out`) has SHA256
`21a58866c2334c3efe54c6db56b4097783b50727b38abdf357b504468989a2dd`.
Independent original source/hash/mode0500 checks pass. Offline
`strata-wide-q6-shaders-20261016.log` exits0 with340 Naga/SPIR-V variants and123
physical-pointer validation checks from that exact frozen executable. The
validator returns before adapter/device creation.

No GPU/adapter/model/inference/service/install/release/API/frontend action. No
physical overlap, numerical/model parity, speed or residency/refund claim. Full
rewrite incomplete; speed gate closed. Next whole-head dispatch, final sampling/
source/queue/ownership/pricing/substitution audit, then full protected numerical/
state/context/hot-request/failure/recovery acceptance under renewed GPU permission.
Preserve previous adaptive substitutions and independent protected-private-caller
boundary; current CLI compiled-public/one-request. Goal active CPU-only; root not
needed; required Vashti changes:none. Canonical/inherited files preserved.

## Whole original target and compact-CJK head dispatch (2026-10-16)

Source `308de4079bae86efa59c9bbeaba75d93dc481448`, isolated `codex/strata-rust-port`. Preserve the
complete branch and owner/admission/graph prerequisites. This supersedes the
whole-head implementation gap recorded in the preceding wide-Q6 receipt.

Pinned MIT upstream `fb58e0dbc8399662c0e47c76578c6e878b14f6cf`:
`sycl/src/core/verify.cpp:1540` and `sycl/src/core/mtp.cpp:1126` submit native
MMVQ over the entire target 248,320-row or original-order CJK 106,299-row head.
The port retains Vulkan's original 16K-row bounded roots and bridges them with
owning immutable source-address metadata, not a merged/repacked weight copy.

- `strata_head::metadata` obtains actual original full-root device addresses and
  byte extents through the same owning client/server stream. It validates exact
  original contiguous page geometry, 4,096-rounded owner padding, actual extents,
  full unoffset distinct handles, 16-byte address alignment and overflow. Its
  initialized 32 KiB table records pointer/byte/row fields with zero unused bytes.
- The new internal vendored metadata method resolves already initialized FULL
  retained same-stream resources on the original Wgpu Vulkan device with its
  existing buffer-device-address feature. It allocates no GPU root, creates no
  device, maps no model/request bytes, executes no GPU work and grants no fence,
  cleanup or admission credit. Unsupported backends reject. It is not an HTTP
  route or external application contract. Original source owners remain retained.
- Actual shared `Weights` owns the target table and whole-head operator; CJK
  `Head` owns its separate table and operator. The target table is allocated under
  existing public persistent/source precharge and completed checkpoint. CJK uses
  original upload/admission/public-guard ownership and exact full-byte startup
  readback for the table as well as its maps. Public CJK enumeration includes its
  table exactly once; cloned retained handles do not create extra physical roots.
- Shared target public bytes become 521,506,816 B (old 521,474,048 + 32,768). CJK public
  inventory gains 32,768 B. The generic Q8 diagnostic MTP public plan remains
  4,203,216,896 B; runtime Q2 public plan becomes 1,946,607,616 B.
  The shared subtraction remains 859,189,248 B for the TWO ORIGINAL GGUF tensors:
  do not subtract either new table as though the sidecar contained it. Combined
  shared model roots gain exactly 65,536 B. Target private 2,778,038,272 B, MTP private
  62,586,880 B and prompt 756/642/32 MiB requests remain unchanged. No physical memory
  recovery/residency conclusion is established by these source prices.
- `strata_dense::whole_head_source` uses the same native wide Q6 math and
  original 210-byte blocks. Its GPU page selector validates actual table row/byte
  fields, then uses local page rows for physical source loads and global rows
  for output writes. All 1–8 RHS variants retain original Q8 planes and reduction
  order. The native physical shader lowering and actual graph use original page
  addresses; ALL original pages are explicit retained read dependencies.
- `GpuExecutor::enqueue_strata_dense` selects shared whole-head execution for the
  actual target output tensor, including the verifier bridge. CJK `Head::project`
  prepares once and enqueues one projection. Groups are 62,080 target / 26,575 CJK,
  within existing 65,535 limits; the final CJK partial subgroup remains bounded.
  The legacy verifier `q6_integer_verify_head_tiles` counter retains its source
  page count 16; it is not a native dispatch counter. Actual dense projection-call
  instrumentation counts ONE whole-head projection. Other formats/indexed Q8
  and feature-off ordinary execution remain unchanged.

CPU receipts under `/media/jmicko/faststore/kuno-development`:

- `strata-whole-head-owned-check-20261016.log`: locked selected release check
  exits0. `strata-whole-head-accepted-test-build-20261016.log`: final selected
  release test build exits0. The earlier `strata-whole-head-initial-test-build-20261016.log`
  is retained as an initial receipt; a source edit during that build required
  the final fresh build. Initial CPU results and both earlier builds are retained
  under `strata-whole-head-initial-*`. The first CPU run caught two fixture
  errors: generic Q8 does not select the CJK table, and the original two-byte
  Q6 scale reads do not call A2. Corrected those exact expectations; no kernel or
  runtime price was loosened.
- `strata-whole-head-cpu-20261016.log` and retained runner: **532 distinct CPU
  cases / 68 nonempty suites, zero failed**, retaining previous 527. New tests
  cover every target and CJK row, original A2 vector bounds including tail
  padding, exact synthetic pointer/table fields and zero unused metadata; reject
  wrong/missing page geometry/owners/addresses; inspect actual full-head WGSL
  descriptors and native physical pointer conversion for BOTH heads and ALL
  eight RHS variants. They do not execute a GPU dot or authenticate real hardware
  addresses. Existing exact shared/CJK/MTP quote fixtures reflect the new roots.
- `strata-whole-head-{core,off}-clippy-20261016.log`: strict locked release
  binary configurations exit0. Changed-source format/whitespace checks pass.
  An additional `--tests -- -D warnings` attempt is NOT green: inherited
  duplicate-module, items-after-test-module and other older test lint findings
  remain in `strata-whole-head-test-clippy-inherited-errors-20261016.log`.
  `strata-whole-head-initial-core-clippy-20261016.log` separately caught a
  modulo-style lint in the new metadata guard; replaced it with the equivalent
  GPU-compatible alignment bitmask before the final build/test run.
  No blanket lint suppression or unrelated test cleanup was introduced. The
  functional CPU test run above is green; do not describe all-target/test lint
  as passing.
- Normal source-clean/stable artifact `strata-whole-head-ordinary-20261016/`
  (`strata-whole-head-ordinary-20261016.out`) has SHA256
  `5766b594472a1debfcf18c07a8ff01459239d60913e3266971a17ba2995a3f0a`. Independent original source/hash/mode0500 checks pass.
  Offline validator `strata-whole-head-shaders-20261016.log` exits0 with 356
  Naga/SPIR-V variants and 139 physical-pointer checks, from that exact
  frozen executable. It returns before adapter/device creation.

Whole rewrite remains incomplete; live speed gate closed. Next final connected
sampling/source/queue/ownership/pricing/substitution audit, then protected full
numerical/state/context/hot-request/failure/recovery acceptance under renewed GPU
permission. Keep adaptive serial-proposal/shared-copy-queue/first-window
substitutions explicit. Protected private caller remains independently gated;
current CLI compiled-public/one-request. No GPU/model/inference/service/install/
release/frontend/API action or physical performance/residency/refund claim.
Required Vashti changes: none. Goal active CPU-only. Root not needed; latest shared
shell refresh still reports a password required. Physical recovery historicalOct 15.
Preserve canonical application work and inherited files. No deployment authority.

## Coupled draft penalty writes and fixed 64 selection (2026-10-16)

Source `8a4ee5b820ad3508b6d7f60d4d58e5efd81bd408`, isolated `codex/strata-rust-port`. Keep the complete
connected branch, including original shared model/arena/graph dependencies. This
supersedes the former draft reuse of target's request-sized inline-penalty sampler.
It does not change which native model, cache format, device or application is selected.

Pinned MIT Strata `fb58e0dbc8399662c0e47c76578c6e878b14f6cf` source audit:

- `sycl/src/kernels/cuda/sampler.dp.cpp:1664..1791` defines `coupled_kpart` as 64
  for both selected vocabularies, and always enqueues original in-place penalty,
  128-thread fixed 64 part, then 32-thread coupled merge. Only merge reads the
  mutable request top-k. Each distinct historical token is penalized once using
  its full window count. The prior Rust reuse fused penalties into part selection
  and used request K for list stride; that source substitution is now removed.
- Actual `strata_sampling_request::draft_pick` invokes `launch_coupled` in BOTH
  full and compact-CJK modes. The actual `coupled_kernels` order is penalty write
  (1024 threads), fixed 64 part (128), merge/request tail (32). Source/kernel IDs
  distinguish these roles and vocabulary. Request values remain in private
  Params and do not enter kernel cache keys.
  The target path still calls its own ordinary `launch`; greedy MTP still uses
  its original argmax/top-probability path, as upstream does when coupling is off.
- `strata-sampler-coupled-penalize.wgsl` declares the original private logits
  descriptor writable. Its complete workgroup membership bitmap is initialized
  before atomic first-writer selection; repeat/presence/frequency arithmetic runs
  once per distinct in-domain history token. History counts use the same selected
  tail; absent compact IDs remain -1 and are ignored. The existing inverse-ID
  bridge maps original history to compact IDs before this stage, preserving each
  count and original-order tie semantics without another weight or history root.
- Coupled part reads already penalized logits with no second penalty and emits
  64 entries per original 4096-logit block, including sentinels for short parts.
  Merge loads/advances fixed 64-stride lists while retaining actual request K for
  the selected tail. Thus request K can change through the original device Params
  without changing captured program identity or misreading candidate layout.
- The existing original candidate owner already covers full 61*64*8 = 31,232 B
  or CJK 26*64*8 = 13,312 B for one actual draft. No GPU root, public/model copy,
  private allocation, source window, INFO owner or admission refund is added.
  All six original private descriptors are retained. Actual native reflection sees
  logits WRITE followed by READ and candidates WRITE followed by READ; existing
  recording dependency tracking orders these original consumers in the same lane.
  Physical execution/order/ownership acceptance is still unproven by CPU checks.
- Audited existing counters: upstream `verify.cpp:2155..2173` draws with
  `Philox(seed,pos0+t)`; `mtp.cpp:1129..1133` uses the same verifying row counter.
  Actual Rust `target_pick` posts counter=start; the original GPU prepare bridge
  uses positions[cell]+1 for each coupled draft, with request/round controls and
  real preceding-token/ring checks. Existing target penalty rows include only
  consumed tail plus the real per-row input prefix; draft rings append only the
  actual pick. Shared full 32K request coordinates and 4096 penalty cap are unchanged.
  `sycl/include/strata/sycl_math.hpp:20..26` selects F32 sampler accumulation by
  default (FP64 requires a separate flag); actual Rust ordered sums/quotients,
  top-p then min-p then temperature, original tie order and Philox are retained.
  Hardware transcendental/numerical parity is not established.

Receipts under `/media/jmicko/faststore/kuno-development`:

- `strata-coupled-sampling-check-20261016.log`: selected locked release check
  exits0. Final `strata-coupled-sampling-test-build-20261016.log` exits0.
  `strata-coupled-sampling-cpu-20261016.log` plus retained runner: **534
  distinct CPU cases / 68 suites, zero failed**, retaining previous 532.
  Two additional cases check actual stage order/read-write descriptors, fixed
  candidate strides and mutable tail K, and independent full-sort results over
  both vocabularies with repeated/absent IDs, positive/negative logits,
  three thread-winner orders, K 1/20/64 and counters crossing 32-bit boundaries.
  Existing sampler source validation lowers all actual new variants. These are
  synthetic CPU/source/compiler checks; they do not run a GPU sampler/model.
- `strata-coupled-sampling-{core,off}-clippy-20261016.log`: strict locked release
  **binary** configurations exit0. Changed-source formatting/whitespace pass.
  Do not claim all-target/test lint is green; inherited findings remain recorded
  in the previous whole-head receipt and were not suppressed or cleaned up here.
- Normal stable-source builder output `strata-coupled-sampling-ordinary-20261016.out`
  freezes the executable under `strata-coupled-sampling-ordinary-20261016/`.
  SHA256 `c8ee12ced38eb658a5ebd8082a221dd0e49ea778b429b78d956417021ea295a6`, independently matched to its original build output and
  verified mode0500. `strata-coupled-sampling-shaders-20261016.log` exits0 with
  360 offline Naga/SPIR-V variants and 139 physical-pointer checks.
  Validator returns before adapter/device creation; no model or GPU was loaded.

Full rewrite remains incomplete, live speed gate closed. Remaining final source/
queue/ownership/pricing/substitution audit and full protected numerical/state/
context/hot-request/failure/recovery acceptance are required. Previous adaptive
substitutions and separate protected-private-caller boundary remain explicit.
No API/protocol/configuration or consumer change; required Vashti changes: none.
Goal active CPU-only; no GPU/inference/service/install/release/frontend action.
Root not needed; latest shared-shell sudo refresh unsuccessful. No physical speed,
recovery, residency or numerical/model acceptance claim. Canonical/inherited work
preserved; no deployment authority.

## Compute-local control uploads and MTP copy scheduling (2026-10-16)

Source `7c35203653d0405e8291bff0307f9438b8376f13`, isolated `codex/strata-rust-port`. Keep the full
connected branch. This resolves the shared-transfer-queue substitution for native
control staging and original lane 1 MTP copies, without adding a hardware queue.
It does not resolve the serial adaptive host proposal or unscoped first-draft path.

Pinned MIT Strata `fb58e0dbc8399662c0e47c76578c6e878b14f6cf`:
`sycl/src/program/generate.cpp:11079..11159` runs adaptation beside accepted-prefix
commit and MTP drafting, joining the host job before the next target window.
Its adaptation uses its own stream; commit/MTP controls are staged on their own
execution streams. MTP's source kernels/step controls use its original compute
stream (`sycl/src/core/mtp.cpp`, `sampler.dp.cpp:1680..1791`). The old Rust global
INFO transfer queue could put both commit and draft controls behind expert copies.

Connected source changes:

- `Graph` owns TWO actual `strata_info_transfer::Uploads`, built for MAIN and
  MTP compute families. `upload_parameters` uses the current original compute
  lane and queue. Actual named replay selects that lane BEFORE staging INFO;
  ordinary scoped producers already retain the same draft-lane authority.
  Target prefix commit contains original GDN/QSA/PLE kernels and dynamic INFO,
  so its control copy no longer queues behind adaptive transfer submissions.
- `Planner::for_lane`, used by the real graph recorder, keeps lane 0's original
  streamed compute/copy schedule and puts lane 1's actual copy/fill/math nodes
  in its original compute stream. This changes only queue assignment: original
  `Dispatch::Copy/Fill`, TRANSFER_READ/WRITE access flags, ranges, release points
  and encoder barriers remain. Compute queues support these transfer commands;
  prompt expert rings and adaptation still use the adopted transfer queue.
  Real MTP recordings have no copy-lane segment/timeline debt. Timestamp metadata
  follows the queues actually used; no profiling or speed trial was run.
- Each actual math flight consumes and credits only its OWN INFO owner's epoch,
  indexed by its original lane, including when a different lane is current at
  join. Scoped completion joins MTP INFO only. Global completion drains both
  owners and their outstanding visibility independently. Queue-fence completion
  never silently credits a newer or peer upload.
- `certify_io_reuse` excludes all genuine source/destination roots of ANY still-
  pending INFO copy, even after the current MTP lane joins. The actual full-owner
  inventory from `Uploads::pending_accesses` feeds the original cohort exclusion.
  A completed draft cannot make unjoined MAIN staging reusable or erasable.
  Cleanup rejects either pending upload owner. Unsupported/failed completion
  retains original owner resources and the existing terminal retirement behavior.
- INFO-only drains still require an actual GPU memory dependency. New
  `Handoff::visibility` records signal-before-wait semaphore submissions even on
  one queue, then joins their original real fence before credit. Ordinary
  same-queue `transfer` retains its no-op behavior; it is not used to invent
  INFO visibility. Cross-lane cleanup additionally propagates to MAIN for Wgpu.
- No model/public/private GPU buffer or hardware queue is added. One additional
  INFO command pool/buffer/timeline/fence is owned through original completion.
  The bounded native queue HOST allowance grows exactly 2 MiB, from 138 to 140 MiB
  (two 2 MiB INFO allowances plus existing 128/4/4 MiB components). The parent
  prompt/controller HOST quote derives from it, growing 266 to 268 MiB. Original
  private roots/source windows and the existing driver/runtime allowance remain;
  physical whole-device/runtime pricing still requires protected acceptance.

Actual static audit findings still requiring follow-up:

1. **Superseded by source `f9c7471f` and the adaptive-host-job receipt below.**
   CPU adaptive ranking/metadata/copy submission previously happened serially before
   commit in `strata_global_moe::begin_adaptation_after_verified`. Pinned serve
   starts `adapt()` on a thread while commit/draft proceeds. Existing code starts
   a GPU copy flight before those consumers; physical overlap and host-schedule
   fidelity remain unproven. Preserve the single serialized submission authority:
   a host worker must not concurrently submit borrowed raw queues or bypass the
   original thread-local private-arena exclusions.
2. **Superseded by source `01291fe0` and the scoped-first-draft receipt below.**
   Final prompt input did use the correct native count1 path and native count1
   state self-commit. However `strata_global_controller::run` prepares its FIRST
   draft outside the scoped pending-commit observer. `Cache.rounds` survives hot
   requests, so adaptation can be due on a later request's first window. Do not
   dismiss the original first-draft substitution as always irrelevant: its
   adaptation currently joins before that outer draft. Later drafts use the
   connected scoped path. Complete this boundary together with retained state.
3. **Superseded by source `39c59c1` and the retained-telemetry receipt below.**
   Retained-session telemetry was not a complete source match. Pinned serve
   initializes `drive.d.usage` before the request loop (`generate.cpp:7626`),
   retains its adaptive rounds and learned `DraftPolicy` (`:8361..8366`), and
   resets token suffix state per request (`:10263`). Rust keeps physical mapping
   and adaptive rounds, but the erased private usage root is not restored by
   `restore_retained_route_map`; `Cache` has no retained usage array. Its native
   controller creates `Policy::new` inside each request. Resolve this specific
   model-local telemetry gap or document a justified privacy substitution before
   closing the source gate. Never retain prompt/token/suffix/KV/request payloads
   or serialize/log per-request traces as a shortcut. General private input and
   hardware hot reuse remain independently unverified.

Validation receipts under `/media/jmicko/faststore/kuno-development`:

- `strata-draft-local-uploads-check-20261016.log`: initial MTP-stage release
  check exits0.
  Final `strata-draft-local-uploads-test-build-20261016.log`: locked selected
  release test build exits0. `strata-draft-local-uploads-cpu-20261016.log` and
  retained runner: **537 distinct CPU cases / 69 suites, zero failed**,
  retaining previous 534. New original-planner tests cover MTP copy/math/fill
  queue assignment, unchanged canonical read/write ranges and release boundaries,
  no invented copy-lane credit, and unchanged lane 0 streaming. New peer test
  composes actual complete-span inventory/cohort exclusion with independently
  joined INFO epochs. Synthetic metadata does not establish driver ownership,
  semaphore visibility, real reuse or physical overlap.
- `strata-draft-local-uploads-{core,off}-clippy-20261016.log`: strict binary
  configurations pass. Changed-source formatting/whitespace pass. No all-target/
  test lint claim: inherited findings remain in the earlier receipt. Initial
  MTP-only checks are retained as `strata-draft-local-uploads-mtp-only-*`; the
  accepted source includes MAIN INFO and explicit same-queue visibility too.
- Normal stable-source artifact `strata-draft-local-uploads-ordinary-20261016/`
  (`strata-draft-local-uploads-ordinary-20261016.out`) has SHA256
  `c2e0c7e52e70ba0d0937eec9e1880f270a96ad366a771e874a4754834f7cc6d0`, independently matched to original build output and mode0500.
  Offline `strata-draft-local-uploads-shaders-20261016.log` exits0 with 360
  Naga/SPIR-V variants and 139 physical-pointer checks. Kernel arithmetic
  and the shader inventory are unchanged; the exact frozen validator returns
  before device creation. No GPU/model/inference or speed test was run.

Full rewrite remains incomplete; live speed gate closed. Resolve the listed
source gaps and final source/queue/ownership/pricing/substitution audit, then
protected numerical/state/context/hot-request/failure/recovery acceptance. Private
caller remains separately gated; CLI compiled-public/one-request. No API, consumer,
configuration, service, install, release or frontend changes. Vashti changes: none.
Goal active CPU-only; no root used. No physical performance, memory recovery,
residency/refund or model correctness claim. Preserve canonical/inherited work.

## Loaded-model usage and draft policy across erased requests (2026-10-16)

Source `39c59c1fc90dc71b94e7f671677fae9c1626be89`, isolated `codex/strata-rust-port`.
This supersedes finding 3 in the preceding compute-local receipt, not its host
scheduling or first-draft findings. Preserve the whole connected branch.

Pinned MIT upstream `fb58e0dbc8399662c0e47c76578c6e878b14f6cf` initializes
`drive.d.usage` before the serving request loop (`generate.cpp:7626`), retains
adaptive rounds and `DraftPolicy` (`:8361..8366`), and resets token suffix state
per request (`:10263`). Earlier Rust kept mapping/rounds but erased usage without
restoring it and created a new policy per request.

Connected owners and behavior:

- Actual `Cache` owns exactly 24,576 aggregate F32 expert counters in a wiping
  vector. Existing adaptive commit retains original decayed scores (same 0.7
  law). These contain no tokens, prompts, seeds, suffixes, KV or identifiers.
- Actual native target reset validates the completed/closed request boundary and
  source readiness, then snapshots the full original 32,768-word usage owner via
  audited I/O. All unused padding must be exact zero; scores must be finite and
  nonnegative. Failed capture retires the original session. No erasure credit is
  invented. The original FULL request reset/inspection still runs unchanged.
- Next request first checks its original target/MTP erasure receipts and retained
  zero cursor. `restore_retained_route_map` checks actual physical placement/cache
  agreement and both original distinct complete 128 KiB controls. It writes only
  routing and bounded aggregate usage, then reads/checks all 32,768 words of each,
  including exact F32 bits and zero padding, under original audited admission.
  No request payload is restored and no expert copy or mapping reset is added.
- Actual loaded `Session` owns the learned `Policy`: fixed aggregate scalar cost/
  acceptance bins, round count and window cap. The controller takes it per request
  and returns it only after original request scopes/fault/I/O closure. Request
  suffix/proposal/token/history owners remain separate and wiping. Invalid window
  bounds cannot consume a prior retained policy. Terminal request failures close
  the session; close drops/wipes policy BEFORE even fallible peer cleanup.
- Model unload/drop wipes the aggregate Cache/Policy owners. Neither has a logging,
  serialization, telemetry export, prompt cache or disk persistence path. This is
  loaded-model scheduler state, not a retained conversation. Current CLI is still
  compiled-public and one-request; private process caller remains independently
  gated. Actual hot-session device acceptance is still required.
- Retained usage is 98,304 B; policy is under its checked 16 KiB bound. Existing
  2 MiB adaptive HOST allowance covers retained arrays, inverse/mapping, snapshots
  and full control staging/readback; no quote or GPU/private/public root changes.
  Snapshot/readback temporaries wipe. Whole-device/runtime pricing remains open.

Validation under `/media/jmicko/faststore/kuno-development`:

- `strata-model-telemetry-check-20261016.log`: initial selected release check.
  Final `strata-model-telemetry-test-build-20261016.log`: locked selected release
  test build. `strata-model-telemetry-cpu-20261016.log` and retained runner:
  **540 distinct CPU cases / 71 nonempty suites, zero failed**, retaining all 537.
  New cases compare an uninterrupted actual adaptive sequence with multiple full
  synthetic request erase/rebind boundaries, exact bits/padding and invalid-score
  rejection; actual learned-policy bins survive multiple request/window boundaries
  without proposal tokens and invalid bounds leave the old owner untouched.
  These are synthetic CPU checks, not live model/state/erasure proof.
- Initial usage fixture wrongly used `Vec::zeroize()` to simulate fixed GPU erasure;
  that empties its length. Corrected cfg(test)-only fixture uses `fill(0.)` and
  retains its fixed extent. Production erase was unchanged. Initial failure and
  build receipts are preserved as `strata-model-telemetry-initial-*`.
- `strata-model-telemetry-{core,off}-clippy-20261016.log`: strict locked release
  **binary** configurations pass; changed formatting and whitespace pass. No
  all-target/test lint claim; inherited test-target findings remain separate.
- Normal stable-source artifact `strata-model-telemetry-ordinary-20261016/`, receipt
  `strata-model-telemetry-ordinary-20261016.out`, SHA256
  `47eb2abef25a3d65e6501b353b27e4f7dc756ee0dcfe7cb51f347b1d030fb823`. Independently matches source
  HEAD, original build output and mode0500. Exact frozen offline validator
  `strata-model-telemetry-shaders-20261016.log` exits0: 360 Naga/SPIR-V variants,
  139 physical-pointer checks. Arithmetic/shaders unchanged; validator exits before
  device creation. No GPU/model/driver/service test was started.

At this historical source, host ranking/submission was serial and final-prompt
count1 drafted outside the scoped observer. Source `01291fe0` below supersedes
that first-draft gap; serial host adaptation remains. Then complete
final ownership/placement/queue/substitution audit and protected whole-model
numerical/state/context/hot-request/failure/recovery acceptance before any speed
trial. Whole rewrite incomplete. No API/configuration/consumer/frontend change;
required Vashti changes: none. Goal active CPU-only; no root used. Shared-shell
sudo refresh was attempted but a usable credential was not confirmed; this block
needed none. No new physical recovery or runtime performance claim. Canonical and
inherited files preserved.

## Scoped first-window MTP consumer and actual runtime nonce (2026-10-16)

Source `01291fe0052ba7e71454ae009b86033972211cd4`, isolated `codex/strata-rust-port`.
This supersedes the outer first-draft gap in the preceding source receipts. Keep
the full connected port. Host adaptive proposal/submission is still serial; this
is not a completion audit, GPU overlap proof or throughput result.

Pinned MIT upstream `fb58e0dbc8399662c0e47c76578c6e878b14f6cf`,
`sycl/src/program/generate.cpp:10995..11159`: first target window is exactly one
final prompt input, does not use suffix lookup or timed policy training, and
still adapts/commits/emits/drafts through the same window path. Retained rounds
can make this first window an adaptive tick on a later request.

Connected execution:

- Actual `prefill_first_native_window` now accepts the SAME higher-ranked
  Verified/Pending observer as later target windows. Its actual full model chunk,
  original verified/fault read, host-row lease release, adaptation acquisition,
  original R pin/readiness submission, count1 candidate state, draft scope and
  final original-family publication remain in the existing transaction.
- Actual controller captures the original MTP binding/cursor before the window.
  Verified records only its real single prediction and opaque original device
  carry. Pending commits the staged sampler input, delivers that prediction once,
  and, only when generation continues, borrows that same verified R and invokes
  actual `prepare_next_draft` inside the original scoped compute lane. EOS/cap1
  skips MTP. Count1 MTP windows still run the genuine accepted front; larger
  windows retain confidence/first-candidate semantics. No invented residual or
  substitute expert/token is used.
- The outer `AcceptedRound` fallback is removed. Every continuing MTP target
  entry now requires the original completed `PreparedDraft` produced inside a
  Pending observer; exact token/cursor/model checks remain. Early first prediction,
  device receipt and original identity metadata wipe/drop after the checked first
  window. First window does not select suffix proposals or train timed policy;
  prompt/output still enter request-local suffix history afterward, as upstream.
- Source audit exposed a prerequisite beyond the callback wiring: actual backend
  `native_submission_scope()` REQUIRES a real reuse nonce, while first prompt
  execution previously preceded the long generation scope. The existing first
  scoped graph preparation would reject that absence. Source evidence, not a live
  device failure, identified this missing owner. No nonce or completion is forged.
- The selected first-window wrapper now validates actual original model/permit
  identity and exact final-prompt cursor, opens the EXISTING construction-admitted
  public-target runtime reuse scope, and enters a distinct `InitialWindow` phase.
  That phase allows only the one final prompt input. It grants neither a generated
  cursor/current generation scope nor completed-request/reset/rearm authority.
  After original model-family acceptance and MTP scope completion, the wrapper
  checks original boundary/fault, calls actual `end_request_io_reuse` with its full
  sanitize/inspection receipt, checks frozen runtime configuration, and only then
  returns to Prefill. Any exception/failure uses original terminal cleanup.
- **Explicit Vulkan ownership substitution:** first-window runtime nonce is a
  short prelude, closed before immutable FourGdn/grouped policy selection and
  the subsequent long generation nonce. Pinned SYCL does not have this Kuno
  runtime-I/O nonce. The actual sanitizer joins original native work, clears
  completed recordings and inspects staging; it does not erase committed model
  private/KV state. No recording from the earlier target policy crosses into the
  newly selected policy's nonce. This adds a real first-window audit/drain; its
  cost and physical behavior are unmeasured, not asserted equivalent/faster.
- Scope counters keep their prior logical completed-request contract: the short
  runtime nonce does not manufacture another closed generation scope. Both actual
  nonces close in sequence, never concurrently. Original HOST/runtime caps,
  private/public/model roots, weights and parameter layouts are unchanged.
  Request metadata is bounded/wiping, no new logging/persistence or API route.

Validation under `/media/jmicko/faststore/kuno-development`:

- Intermediate `strata-first-scoped-draft-{check,scope-check}-20261016.log`
  compile callback and initial runtime-scope wiring. Accepted final source uses
  explicit enum `InitialWindow` and actual phase-restricted end/current-scope.
- Final `strata-first-scoped-draft-test-build-20261016.log`: locked selected
  release test build. `strata-first-scoped-draft-cpu-20261016.log` and retained
  runner: **542 distinct CPU cases / 71 suites, zero failed**, retaining all 540.
  New actual authority cases cover prompt lengths1/35/4096/32767, exact count1,
  missing/mismatched actual nonce, cursor/phase bounds, wrong close/reset/rearm,
  no generated scope-number authority, and retained sequential requests.
  This validates CPU owners/guards, not actual callback/math/device completion.
- Initial test exposed an actual transition bug: extending active runtime state
  with an extra boolean allowed generation `end` to accept the short scope.
  Replaced it with explicit enum phase and strict generated end/current-scope.
  Initial receipt preserved as `strata-first-scoped-draft-initial-*`. Its runner
  also correctly rejected a redundant child filter already covered by its broad
  parent; the final runner has no duplicate and includes both new cases there.
- `strata-first-scoped-draft-{core,off}-clippy-20261016.log`: strict locked
  release binary lint passes selected Intel and ordinary Strata-feature builds.
  Initial off invocation omitted the binary's REQUIRED `strata-rust-port` feature
  and failed before compilation; preserved as `initial-off-clippy`. Corrected
  off receipt explicitly enables that feature without Intel. No all-target/test
  lint claim. Changed Rust formatting and whitespace pass.
- Normal stable-source artifact `strata-first-scoped-draft-ordinary-20261016/`,
  receipt `strata-first-scoped-draft-ordinary-20261016.out`, SHA256
  `8ced9ff4099d1a4f2fc3f47e6880af9e92d03696d3e6e0cc6bd4b50e9eb42c42`. Source commit, original build
  output, digest and mode0500 independently match; frozen binary not executed.
  No shader/kernel arithmetic or inventory changed. The preceding `39c59c1`
  frozen artifact's offline 360 shader variants/139 physical-pointer checks remain
  the latest shader receipt; not rerun here and not new GPU execution evidence.

Remaining: serial adaptive host ranking/submission; final whole-port source,
placement, queue, ownership/pricing and substitution audit; protected whole-model
numerical/state/context/hot-request/failure/recovery acceptance; then speed tests.
Complete rewrite remains incomplete. CLI compiled-public/one-request; protected
private process caller independently gated. No API/configuration/consumer/frontend
change; required Vashti changes: none. Goal active CPU-only; no GPU/model/inference,
service/install/release/root action. No privileged credential was needed or used.
Canonical/inherited files preserved. No new physical recovery/performance claim.

## Adaptive host job and owned transfer submission authority (2026-10-16)

**Historical extension receipt:** its GPU-derived selected usage and universal12MiB
allowance are superseded by the selected NO_HOST policy receipt below. The actual
promotion/copy worker remains available explicitly; its ownership evidence stays
valid for that extension, not as a claim that pinned NO_HOST promotes experts.

Source `f9c7471f498e877be1a9c733aa1a3db3ccdfb154`, isolated `codex/strata-rust-port`.
Keep the full coupled branch. This supersedes the serial host-job gap; first
count1 scoping (`01291fe0`) and model-local usage/policy (`39c59c1`) remain selected.
This is source/CPU acceptance, not full rewrite or device/performance acceptance.

Pinned MIT Strata `fb58e0dbc8399662c0e47c76578c6e878b14f6cf`,
`sycl/src/program/generate.cpp:7707..7883` and `:11079..11159`: adaptive selection
and copies run on a host thread beside target commit/MTP; the loop joins that job
before proceeding. Eligibility >=2, gain1.5, decay0.7, every4 and maximum96 swaps
remain the existing connected Rust Cache rules. No speculative expert substitution
or speed-driven rollback is introduced.

Actual connected path and ownership:

1. Actual Verified target rows/fault are observed and their host-row lease drops.
   The selected adaptive tick reads the original bounded GPU usage root, moves
   the SAME Cache and wiping usage vector into a real host job, and returns a
   `Flight` before original target commit/MTP proceeds. No model client or arena
   is moved to the worker. This applies to the scoped final-prompt count1 too.
2. Sources build an authenticated, immutable 48x512x3 physical-cell inventory ONCE
   during original model load. It resolves original GPU/HOST handles via the SAME
   client and plan, checks full resource identity, exact ranges/alignment/copy flags
   and bounded metadata, and retains Buffer aliases rather than copying weight
   bytes. Cells are original physical addresses, not current logical expert IDs;
   the learned permutation still chooses their occupants. Normal source close
   drops this inventory before releasing original HOST/GPU owners.
3. On the sole client thread, Source checks Cache/map agreement, joins original
   readers, verifies the complete attached original32MiB prompt raw scratch,
   acquires its original same-thread external-flight exclusion and revokes all
   genuine HOST EXPERT pages before selection. Embedding/shadow pages stay separate
   so MTP's genuine immutable inputs remain active. Rebase token/client and actual
   allocator/root/TLS guard stay in the parent !Send Flight. A failure before any
   scratch work releases only that known-idle root; source mutation stays terminal.
4. Parent creates the original transfer objects and enqueues the real MAIN-ready
   signal BEFORE spawning. `spawn_host_job` owns a bounded2MiB stack. The worker
   uses the SAME unchanged `Cache::propose`, validates original physical pair
   types, equal role sizes and disjoint spans, records original three-role
   HOST->scratch / GPU->HOST / scratch->GPU copies and their RAW/WAR/WAW/HOST_READ
   barriers, and submits only on the actual adopted TRANSFER queue. Worker gets
   owned Buffer/device/queue aliases plus numeric metadata, NEVER a ComputeClient,
   registry token, arena, allocator capability or MAIN queue. Its raw command
   objects cannot be destroyed until the parent has joined the real host thread.
5. New HAL `KunoTransferQueue` keeps a strong Arc to the SAME actual DeviceShared
   and that queue's one submission mutex. Both actual native `Commands::submit`
   copy segments and worker copies use it; old borrowed getter is removed. Native
   graph validates device/queue identity before submission. Compute/INFO/MAIN
   remain on the original serialized engine thread. The lock covers only actual
   `vkQueueSubmit`, NEVER a fence wait or thread join. Natural Rust Send/Sync is
   compile-checked; no unsafe Send/Sync is added for borrowed queues/arenas.
6. After original commit/MTP and family consumers, parent joins its real worker.
   Worker completion/AtomicBool records only ATTEMPTED copy debt. Only actual
   original ready/copy fence success grants their completion; parent then submits
   and observes real copy->MAIN semaphore/fence visibility. No worker submits
   MAIN. Unknown completion retains original buffers/inventory/device and leaves
   the original root blocked/quarantined. Drop joins before object/arena teardown.
7. Only successful worker/copy/visibility completion returns the SAME Cache and
   proposed map. Parent validates next physical placement, writes original full
   private map/usage controls, performs the existing actual client join, commits
   Cache and publishes source placement. No next target can see half a swap.
   Empty-swap/all-resident cases issue no copy and need no invented copy-fence or
   HOST-rebase completion; missing token cannot authorize a nonempty swap set.

Explicit bounds/substitutions and open physical proof:

- Adaptive HOST allowance grows **2->12MiB**: existing2MiB control/selection
  temporaries +2MiB worker stack +8MiB bounded full inventory. Existing model/frame
  HOST pricing derives from this constant; quote/cap tests pass. Original source
  weights, private/public GPU roots,32MiB raw scratch, driver/queue allowance140MiB
  and prompt/controller268MiB quote are unchanged. No weight mirror or extra GPU
  queue is added. Physical whole-device/runtime/HOST pricing remains unverified.
- Rust/Vulkan uses the already adopted transfer-family queue with shared owning
  host serialization, not a fourth hardware queue/borrowed raw handle. MAIN-ready
  submission, authenticated inventory and revocation are prepared on the client
  thread; ranking/recording/copy submit is the actual concurrent host job.
  Scheduling opportunities exist in source; independent GPU overlap and cost
  require protected physical acceptance. An implementation-time speed number
  would not justify disabling this coupled path.
- Source privacy is unchanged: no prompt/token/seed/suffix/KV/output value is passed to or read by the
  host job; Cache/usage/proposal numeric owners wipe, inventory is immutable PUBLIC
  model geometry, all request arenas still erase in full. No logging, persistence,
  telemetry export, API route or consumer/configuration change. Current executable
  is still compiled-public/one-request; private process caller is separate.

Validation under `/media/jmicko/faststore/kuno-development`:

- `strata-adaptive-queue-authority-check-20261016.log`: initial owned-queue release
  check. Intermediate `strata-adaptive-host-worker-{check,owned-check}-*` found
  incorrect Option handling of HOST role metadata; fixed before connected check
  and final acceptance. `strata-adaptive-host-worker-connected-check-20261016.log`
  compiles the connected job. These are intermediate, not accepted source hashes.
- Final `strata-adaptive-host-worker-test-build-20261016.log`: locked selected
  release test build, no source warning. Final `strata-adaptive-host-worker-cpu-20261016.log`
  and retained runner: **545 distinct CPU cases / 71 suites, zero failed**, keeping
  all542. New cases use the actual bounded thread launcher and real Cache to prove
  a live worker can proceed beside foreground work and yields the same proposal,
  join/classify errors/panic, and prove host completion grants only attempted
  submission debt, not copy or MAIN visibility completion. Existing role-copy,
  queue/dependency, private exclusion, placement, scope and retained-state cases
  remain. Synthetic CPU work is not actual GPU/registry/allocator acceptance.
  Initial545 run/test build is retained as `initial-*`; final source additionally
  releases known-idle scratch on pre-worker registry failure and removes the
  resulting unused-mut warning. No coupled behavior was rolled back for timing.
- `strata-adaptive-host-worker-{core,off}-clippy-20261016.log`: strict locked
  release **binary** builds pass Intel/core and ordinary Strata-feature configs.
  All 11 changed Rust files format; whitespace passes. No all-target/test lint claim.
- Normal stable-source artifact `strata-adaptive-host-worker-ordinary-20261016/`,
  receipt `strata-adaptive-host-worker-ordinary-20261016.out`, SHA256
  `d4cfebbb5a28d645b296c20aaa4dea09239b961f1cea4456ab8299ed2e17e7e5`. Source, original build digest
  and mode0500 independently match. Frozen binary NOT executed. Shader arithmetic/
  inventory is unchanged; prior `39c59c1` frozen offline360/139 remains the latest
  shader receipt, not rerun and not proof of new driver/queue behavior.

The three named concrete gaps from the compute-local audit are connected, but
**completion remains unproven**. Next do the full branch's final source/placement/
queue/ownership/pricing/substitution audit against pinned selected Intel serving,
then protected numerical/state/context/hot-request/failure/recovery acceptance,
then live speed tests. Private/general caller independently gated. Goal active
CPU-only. No GPU/model/inference/service/install/release/frontend/API/root action;
Vashti changes: none. Canonical/inherited files preserved. No sudo needed/used.
No new physical recovery, GPU overlap, model correctness or throughput claim.

## Selected NO_HOST policy and native router/plan schedule (2026-10-16)

Runtime sources `3d6a0390815e0ac116f98d765c398e5b74b1cec9` (policy) and
`0d9e3a527041c36b2a395cda5c1070a88288684d` (selected router/planner), isolated
`codex/strata-rust-port`. This is final source-audit correction, not a speed-based
reversal or full rewrite acceptance. Keep the whole coupled branch.

Pinned reference `fb58e0dbc8399662c0e47c76578c6e878b14f6cf` is still verified.
The selected launcher `sycl/serve/strata-sycl.sh:22..25` defaults DEVICE_PLAN=1,
NO_HOST=1. `expert_source.cpp:2578..2580` increments usage only in the expert
callback, and `verify.cpp:1956..` skips that callback loop in NO_HOST. Therefore
selected counters remain zero. `generate.cpp:11086..11159` still starts/joins the
adaptive host job every4 windows beside commit/MTP; zero counters yield no swaps.

Actual Rust policy connection:

- `verified_window` controls logical tick/job tracking independently from actual
  GPU collection. The shared policy helper supplies BOTH cold/captured INFO word3
  and retained native replay RouteUsage; original completion guard checks that
  same value. Reference mode sets zero everywhere, including the final-prompt
  count1. Actual resident and diagnostic callers use the renamed window setter.
- A real bounded `SelectionFlight` owns SAME Cache/wiping zero counters, runs the
  unchanged proposal on the2MiB host stack, and joins before publication (also on
  Drop). It validates no swaps, zero stats and unchanged map, then commits decay.
  It owns no client, registry/arena/scratch capability, GPU buffer or copy command.
  No GPU counter readback, copy fence, registry revocation or unchanged-map upload
  is invented at an empty tick. Request-end full stats audit, full arena erase and
  original route/usage-root rebind audit remain; retained policy remains loaded-
  model-only and request tokens/suffix/KV still wipe.
- Prior authenticated inventory / actual rank-record-copy worker is preserved in
  `strata-gpu-adaptive-usage`, which implies Intel core. Builder requires explicit
  `--intel-core --gpu-adaptive-usage`. Normal `--intel-core` excludes that feature.
  Default does not allocate its unused8MiB inventory. Adaptive HOST allowance is
  4MiB reference (2 control+2 stack),12MiB extension (+8 inventory). This is faithful
  mode selection, not a disabled upstream dependency based on a slow trial.

Selected router/planner connection:

- `native_router.dp.cpp:70..166,269..286` actually uses one32x8 block PER token,
  only row zero active. Rust decode now launches count256-thread blocks and keeps
  one active32-lane subgroup each, with no divergent workgroup barrier. All16
  logits/lane, XOR16/8/4/2/1 normalization/top10 reduction, stable ID ties,
  winning-expert-lane sums and2^-14 clamp follow that source. It writes logical
  IDs in the last80 words of the SAME8192-word private map root and original
  probabilities. Non-finite input still classifies a terminal fault.
- `verify_kernels.dp.cpp:2062..2217,2361..2382` uses128 entries/4 subgroups and a
  packed `(entries<<16)|groups` prefix scan. New `strata_global_plan.wgsl` ports
  that scan, first-occurrence group order and member rank. A separate selected
  native graph node reads the original logical IDs/remap, writes the original
  fixed8 member planes, and preserves all genuine expert selections. Both nodes
  bind the SAME map/fault roots; reflected read/write accesses create original
  capture/replay RAW/WAW dependencies before actual source-table role consumers.
  Both carry genuine epoch/usage INFO fields. Kernel metadata is256/128 as the
  WGSL declares. Prompt's separate large-chunk route/group path is unchanged.
- Vulkan substitution remains explicit: authenticated physical cell IDs and the
  original role address tables replace raw pointer pairs in the plan, and the
  existing fixed8 group member layout replaces packed offsets. Full original
  source partition validation happens before decode. No expert weight mirror,
  additional private GPU root, CPU expert fallback or approximate route is added.
  Native INFO adds one16-byte record per routed layer within existing bounded
  template/page owners; runtime capacity and physical pricing still need whole-
  model acceptance. WGSL exp and compiler arithmetic are not a bit-parity claim.
- Earlier preliminary concern about a different per-token reduction was corrected
  by reading the active subgroup: BOTH versions used one subgroup/token. The
  actual remaining schedule difference was fusion and serial group construction;
  this block ports the selected separate launch/parallel scan, not invented math.

Reference configuration evidence: saved public comparison options select spec4,
context32768, prefill4096. Current native request already selects `Config::native(4)`;
MAX8 is capacity. Both original public expert profile and Rust included fixture
have SHA256 `8f59b4aa8873209dff11c11e37bcda9529a1335b724a1afeea37bf6388975baf`.
No request content or credentials are copied into these notes.

Acceptance receipts in `/media/jmicko/faststore/kuno-development`:

- Policy source: `strata-final-audit-usage-{test-build,cpu,core-clippy,off-clippy}-20261016.log`:
  547 cases/71 suites, zero failed; strict locked release binary lint in core/off.
  `strata-final-audit-usage-optin-cpu-20261016.log`:6 actual focused CPU cases pass
  under extension compilation. Initial missed diagnostic setter callers and
  large-enum lint were fixed (original copy Flight boxed); initial receipts stay
  explicitly intermediate. Builder AST selector checks pass. Normal frozen source
  `3d6a0390`, artifact directory `strata-final-audit-usage-ordinary-20261016/`, SHA256
  `51633435b109c09f120319edc190adeab990ca7ac752bf7eccd96f19e5240e2e`; independent
  source/original hash/mode0500 and exact default features pass. Not executed.
- Router source: `strata-router-plan-{check,targeted-cpu,cpu,core-clippy,off-clippy}-20261016.log`:
  549 distinct CPU cases/72 suites, zero failed, retaining all547. New cases compile
  actual route/plan WGSL through Naga/SPIR-V and check metadata/descriptors/root
  bounds; an independent stable-group oracle checks the packed parallel scan,
  duplicate-free output writes,1..8 windows, four expert patterns and three full
  physical permutations, including the7/8-token tail and all subgroup boundaries.
  Strict locked release binary core/off lint and changed formatting/whitespace pass.
  CPU/compiler checks do not establish GPU arithmetic or actual driver scheduling.
  `strata-router-plan-optin-cpu-20261016.log` additionally passes the six retained
  usage/host-job cases and both router/planner cases under the actual explicit
  `strata-gpu-adaptive-usage` feature on this same source (8 cases, no GPU).
- Normal builder artifact `strata-router-plan-ordinary-20261016/`, receipt
  `strata-router-plan-ordinary-20261016.out`, source `0d9e3a52`, SHA256
  `2d60cec93970727aeea74a1d7ad3f362cd98dc2ea300e56e62f1a79d94c39a8c`. Independent original/frozen hashes, mode0500,
  source and exact features `[strata-rust-port,strata-intel-core]` match. Only frozen
  `--validate-port-shaders` ran (returns before adapter/device initialization).
  `strata-router-plan-shaders-20261016.log` passes361 WGSL/SPIR-V variants,
  139 physical-pointer checks and32 matrix configurations. Matrix-aware validator
  PATH is `spirv-tools-matrix-20261014/build/tools` under the receipt directory,
  pinned tool source `fbe4f3ad913c44fe8700545f8ffe35d1382b7093`. The initial system
  validator lacked BF16 capability5118; its failure is preserved as
  `strata-router-plan-system-validator-initial-20261016.log`. Corrected PATH passes
  without engine/source changes. No adapter, model or inference run.

Full rewrite remains incomplete. Finish the complete branch's source/ownership/
placement/queue/pricing/substitution review, then protected whole-model numerical,
state/prefix acceptance/rejection, context, hot-request, failure and actual resource
recovery acceptance. Private process caller remains separately gated. Only then
open the speed gate. No live GPU/speed/model/service/install/release/frontend/API
work here; required Vashti changes: none. Goal active CPU-only. Canonical checkout
and inherited untracked cache preserved. Sudo refresh was attempted noninteractively
but no usable credential was confirmed; no root operation was performed.

## Native context headroom and full-window scope audit (2026-10-16)

Runtime source `c5fcc20bc44e8dce900ac3bc4d1865477854f2fc`, isolated `codex/strata-rust-port`, after
`0d9e3a52`. This block is engine inference/state fidelity, not gateway/auth/UI work.
Keep the complete coupled port; no timing-driven selection or rollback.

Pinned Intel serving source `generate.cpp:9406..9410` rejects a request unless
`prompt + max_new + 8 <= max_context`. Its serial loop `:10993..11034` keeps the
chosen full window instead of clipping at remaining delivery count. `:11084..11140`
commits `a+1` INPUT rows, but emits only through EOS/output cap and runs a next
MTP draft only when continuing. Thus a final accepted target cursor can exceed
the cursor suggested solely by the number of delivered output tokens. The extra
cells are runtime headroom, not additional outputs.

Connected Rust correction:

- `strata_engine::Request`, its actual reader/tokenizer bound, shared `request_end`
  helper and `controller::Request::validate` now preserve eight cells BEFORE model
  inference. Largest one-token-prompt output count is32759 at32768 context. No
  truncation, seed/sampler rewrite, smaller verification window or hidden output.
- Actual `resident_language_request_io::Geometry` and original `Authority` constructor/
  retained rearm validate that same geometry and its admitted physical headroom.
  A request valid only by consuming the headroom cannot enter an original scope.
- For the selected core with four-row/native-window authority enabled, the LAST
  continuing origin remains strictly below `final_cursor=prompt+outputs-1`.
  Its complete verification/accepted commit may end as high as
  `final_cursor+verify_capacity()-1` (MAX8=>+7), within the original physical capacity.
  `run`, active close and closed-request/reset checks use the same bound. No new
  generation may start at or after the delivery-final origin. EOS/cap never clips
  actual acceptance merely to manufacture the old cursor.
- Original short initial count1 scope, phase/counter/model identity, actual target
  position, family commit/fault/I/O completion, erasure receipts and next-request
  rebind remain independent requirements. Bounds are NOT completion/release proof.
  Historical feature-off fixtures and non-native width<=3 guards retain their
  former bounds. No GPU root, arena size, storage format or external API changed.

Validation receipts under `/media/jmicko/faststore/kuno-development`:

- `strata-completeness-context-test-build-20261016.log`: final locked release
  selected test build; `strata-completeness-context-full-cpu-20261016.log` and
  retained runner: **551 distinct CPU cases/72 suites, zero failed**, retaining
  all549. Two added cases check literal eight-slot serving admission across edge
  partitions/overflow and the actual original scope through every window1..8,
  remaining-delivery1..8 and accepted-prefix1..width, including commit beyond
  delivery, insufficient admitted capacity, closure/reset bounds and refusal to
  start another window at the delivery cap. Updated existing fixtures still
  prove original short scope/rearm/legacy guards. Final
  `strata-completeness-context-{core,off}-clippy-20261016.log`: strict locked release
  **binary** lint passes; three changed Rust files format and whitespace passes.
  This is CPU/state-contract evidence, not actual target/MTP/GPU acceptance.
- Initial engine-only13 cases passed, then the broad runner exposed old maximum
  output fixtures. Reviewing their actual scope guard also exposed its coupled
  full-window rejection. Constructor/rearm, active/closed bounds and tests were
  corrected together. Initial failing receipts are retained as
  `strata-completeness-context-initial-full-cpu-20261016.log` and
  `strata-completeness-context-scope-initial-cpu-20261016.log`, not acceptance.
  The pre-feature-guard551-case receipt is also retained separately; final tests
  rebuild the actual final source with non-core test gating corrected.
- No new normal frozen artifact was made for this intermediate audit block.
  Latest frozen normal build/shader acceptance remains explicitly source`0d9e3a52`
  in the prior receipt; its binary does NOT include this context/scope correction.
  Changed source has no new shader; no repeated offline shader or live GPU run.

### Next required coupled block: selected shared expert

This is a concrete source dependency, not an optional speculative optimization.
Pinned `verify.cpp:100..116` defaults `sh_stream_on()` true outside HIP; the SYCL
build does not define HIP, and recorded comparison config has no override.
`:1212..1224,1284..1322,1415..1424` forks the independent shared stream and joins
its event before combine. `sh_fork_late` defaults true, so selected DEVICE_PLAN /
NO_HOST mixed-cache work can queue the router/plan before forking shared work.
The current Rust selected loop `resident_model_batch.rs` enqueues routed expert
roles, `self.shared.prepare`, route reduction and `self.shared.finish_prepared`
all on MAIN. There is no shared compute lane/tag in original Graph Node; native
`strata_schedule::Planner` tracks only compute/copy. Independent MTP compute is
already real, but does not by itself implement a shared fork inside a target graph.

Port these dependencies as ONE connected block before judging speed:

1. **Selected math and source switches.** Native preset sets native BF16/extra at
   `generate.cpp:2251` and shared BF16 selection at`:2803`.
   `shared_expert.dp.cpp:304..383` defaults fused SwiGLU->Q8_1 ON, projects gate/up
   from the authentic Q8 image, consumes the fused packed image in DOWN, computes
   the scalar gate with original native BF16 GEMV (single or multi), then performs
   rounded sigmoid scaling. Current generic64-lane `resident_shared_batch::gate_dot`
   is not that producer. Reuse audited Rust native dense/BF16/Q8 primitives where
   they preserve the original math and owners; connect actual native shared
   consumers. Audit count1 and multirow combine dispatch/order in `native_moe.dp.cpp`.
   Do NOT assume CUDA feature switches apply: `STRATA_LFUSE` and VERIFY_QDEDUP /
   QFUSE default OFF here. Gate/pair subflags alone do not enable LFUSE. No fused
   deferred-gate/router requirement unless the selected master switch is on.
2. **Independent live scratch and faults.** Audit actual `ResidentBatchProjection`
   and `dense_q8` owner against routed-expert GU/DOWN packed roots, shared
   GATE/UP/ACT/DOWN/SCALE and next-layer reuse. Never add only a queue switch while
   sharing destructive scratch. All current kernels also write the same atomic
   global fault root: honest RAW/WAR/WAW planning would serialize that fork.
   Use an owned one-logical-word shared fault and a real joined fold into original
   global fault before final head/family acceptance, or prove an equally faithful
   atomic protocol. Reserve/erase/audit/charge that owner. Do not silently weaken
   all fault hazards. `outer.packet` has MAX_BATCH*512+1 LOGICAL words, so it cannot
   simply be passed as the single-word fault ID (`enqueue_batch` checks words==1).
3. **Original graph and device queues.** Extend actual Node/planner/recording/
   command submission so a shared section inside a target branch really uses a
   separate COMPUTE-capable queue with exact fork/RAW/WAR/WAW/join waits. Record
   scope/tag on genuine enqueue, including cold and retained replay; do not infer
   completion from a source name or bool. Existing compute lane1 can potentially
   host target shared work and later MTP because selected target verification
   joins BEFORE MTP drafting, but this is a proposed Vulkan substitution requiring
   explicit owner/flight/INFO/readiness proof. No simultaneous alias or borrowed
   raw queue may bypass the existing owning authority. Never switch the whole
   current Graph lane merely to tag a subsection (that drains and destroys its
   scratch pools). Dedicated transfer queue is not assumed compute-capable.
4. **Complete submission and cleanup.** Generalize real segment/timeline/family-
   specific pools and all-resource access tracking, final owning fence, cached
   recording identity, INFO visibility, metadata/driver/HOST pricing and failure
   quarantine together. A parent target fence must join all shared/copy work;
   later MTP must not overlap unfinished shared work on a reused queue. Retain
   all original roots/objects through unknown completion. No host readback/new
   per-layer submission fence should replace the original GPU fork/join.
5. **Acceptance.** Keep all source pieces connected under selected Intel core.
   Test CPU scheduling hazards across all queues, fork input visibility,
   disjoint overlap, original combine/fault dependency, replay epochs and final
   all-lane drain; compile actual kernels and exercise independent numerical
   controls. Only then protected count1/multirow whole-model correctness/recovery.
   No speed ranking until the full source and execution gates pass.

This block has not implemented that shared fork; mark it missing rather than
claiming the existing MTP queue completes it. Continue other full-branch checks
as necessary. Required Vashti changes: none. Goal active CPU-only. No root,
GPU/model/inference/service/install/release/API/UI action, no new physical recovery
or throughput claim. Canonical/inherited files preserved.

## Connected shared expert and three-queue target graph (2026-10-17)

Runtime source `46e695a7c6462eab86171086d04c742b3e4386de`, isolated `codex/strata-rust-port`, following
`c5fcc20b`. This is the coupled block required by the preceding context/shared audit.
Keep the entire branch; this is not a throughput-selected candidate or rollback.
Pinned MIT reference remains Strata `fb58e0dbc8399662c0e47c76578c6e878b14f6cf`.

Source mapping and connected behavior:

- Selected target/count1..8 now queues router/plan, enters a shared COMPUTE
  section, then queues routed roles on MAIN. An explicit MAIN segment release
  separates those roles from the later shared-fault fold/final combine. Without
  that release, aggregating the later join into the same segment would serialize
  the supposedly independent routed work. CPU-only host enqueue boundaries are
  not Vulkan fences or readiness/acceptance receipts.
- `resident_shared_batch::prepare_native` reuses the authentic gate/up cached
  quantized matrices and existing dense Q8 scratch: pack the real GR input once,
  project gate and up from that image, then fuse original
  `(gate/(1+exp(-gate)))*up` with the native subgroup Q8_1 producer. DOWN consumes
  those already packed planes without re-quantizing the obsolete ACT root.
  Native 640-column shape is20 blocks,360 plane words per active member. The
  packed image is independent of global routed GU/DOWN roots. Full physical
  alias checks include scratch and every original shared/input/output/fault owner.
- The scalar gate is the authentic original rank-one BF16 tensor. A dedicated
  projector seam uses the already-existing `strata_projection_shape` and sealed
  native BF16 cache, not `Matrix::new` (which correctly rejects rank1), a generic
  64-lane dot, or reinterpreted/new weights. Native128-thread vec4 sigmoid-scale
  stores the separately rounded shared row. The actual128-thread vec4 combined
  reduction starts with a rounded rank0 product, ordered FMA ranks1..9, and then
  adds the scaled shared row, preserving original role/status/map/fault/stamps.
  Existing prompt GEMM/shared and non-core generic paths are separate, unchanged
  algorithms; this block does not assert an additional prompt-stream port.
- Shared kernels write one separately admitted logical fault word (a complete
  32KiB private owner). A MAIN fold genuinely reads it and ORs into the SAME model
  fault before the combined result is accepted. General fault hazards are not
  weakened. Shared preparation/consumption stays single-use and sticky-poisoned
  on failure. Every original shared root remains covered by full arena erasure
  and retained-replay validation; the unused ACT owner is intentionally retained.
- Actual vendored Runtime client/server/stream APIs drain host tasks before tagging
  genuine graph Nodes. Stream validates the active private request nonce and
  actual target branch. Nested/foreign/end mismatches, MTP use and branch changes
  inside a shared section reject. The RAII enqueue scope attempts End on unwind;
  End is never GPU completion. Only the original all-queue drain can clear a
  leftover graph tag during cleanup.
- The native scheduler tracks three relative lanes: primary, original owned COPY,
  and shared COMPUTE. Reuse existing adopted compute queue1 for shared target
  work, then later MTP. No fourth queue or compute use of a transfer-only queue.
  Each real descriptor/physical/copy byte interval retains RAW/WAR/WAW history
  across all three lanes. Each segment carries ALL peer timeline waits, not a
  scalar maximum that loses a peer. Initial header transfers original INFO debt;
  final target fence waits BOTH last-copy and last-shared timelines even with no
  later consumer. MTP copies remain inline on their own primary queue; a shared
  tag there is forbidden. Target completion precedes reuse by scoped MTP.
- Each shared command buffer uses its actual compute family's separately owned
  pool. Actual encoding has three per-queue access inventories; COPY/FILL on the
  shared compute queue retain real TRANSFER operations. Queue tags and explicit
  releases are in original recording signatures, so cold/capture/replay all use
  the same schedule and retained resources. Existing unknown-submission failure
  quarantine, original target fence/flight and owned-copy submission authority
  remain. Timestamp profiling, if explicitly selected later, serializes all three
  queues and uses their common supported timestamp width; no timing evidence here.

Admission and substitutions:

- Target shared private pages grow262144->294912 bytes for the additional fault
  owner, reflected by both the shared allocation request and actual parent Plan.
  No extra model weights, expert mirrors or request content persistence.
- Native queue HOST allowance base128->144MiB (+16MiB), propagated through the
  existing parent prompt HOST quote. The added third hazard lane costs16 bytes
  per reserved Region (at most524288=>8MiB); larger segment arrays, third queue
  tracking, signatures and pool ownership fit the conservative remainder. Region
  size64/Segment<=48 and capacity bounds are CPU checked. Actual allocator/driver
  consumption still needs protected device admission acceptance; this allowance
  is not a measurement or claim about physical overlap.
- Reusing the existing MTP-capable compute queue, fixed8 member planes, Vulkan
  pools/timelines and explicit fault fold are documented backend substitutions.
  No fake receipt, CPU expert callback, shortened verification or throughput
  comparison selects this path. Original reference master-switch defaults from
  the preceding audit still apply (no LFUSE/QDEDUP/QFUSE forced on).

Validation receipts under `/media/jmicko/faststore/kuno-development`:

- `strata-shared-connected-final-test-build-20261017.log`: final locked release
  selected test build. `strata-shared-connected-full-cpu-20261017.log`: **558
  distinct CPU cases/73 nonempty suites, zero failed**, retaining all551 prior
  cases. The actual vendored scheduler CPU module includes the four new shared
  cases in its existing broad suite (no duplicate suite needed). New cases check
  independent routed/shared overlap versus the late join, all peer timelines,
  three-lane pairwise byte hazards, original shared COPY/FILL tagging, MTP
  exclusion, metadata bounds, authentic fused producer lane/plane ownership and
  actual Naga/SPIR-V shader bindings/workgroups/native ordered combine interface.
  Retained runner is `strata-shared-connected-validate-20261017.py`.
- `strata-shared-connected-{core,off}-clippy-20261017.log`: strict locked release
  **binary** lint passes; changed Rust formatting and whitespace pass. Initial
  test build exposed two old `NodeAccess` fixture constructors needing the new
  queue field; corrected all consumers together. Initial full CPU log is retained
  separately: the new WGSL `shared` identifier was reserved and was corrected to
  `shared_result`. The runner also rejected a duplicate shared-suite filter; the
  existing broader filter already ran those four cases, so removed the duplicate
  and added only the new native shader suite. No failing initial receipt is
  counted as acceptance. Final pass rebuilt the corrected selected source.
- `strata-shared-connected-ordinary-20261017/` is the normal stable-source frozen
  build, receipt `.out`, source`46e695a7c6462eab86171086d04c742b3e4386de`, SHA256
  `7834bf0acc9d8e6e02fd53f0f9d198e8f0aefdc980206182c8203c06847ee20a`. Independently checked current source/original binary/frozen hash,
  mode0500 and exact default Intel features (no GPU-usage extension). Its ONLY
  execution was `--validate-port-shaders`, before vendor/adapter/model setup:
  **365 shader variants/139 physical-pointer checks/32 matrix configurations**
  pass in `strata-shared-connected-shaders-20261017.log`. Pinned matrix-aware
  SPIRV-Tools validator `fbe4f3a` used via its existing tools PATH; no runtime engine
  dependency added. New variants are fused shared Q8, native scale, fault fold
  and joined vec4 combine. No actual GPU API/adapter/model initialization,
  inference, numerical/overlap/physical memory/recovery acceptance or speed trial.


Full rewrite and speed gate remain incomplete. Next: final connected branch
source/placement/queue/ownership/pricing/substitution audit, then protected actual
count1/multirow numerical, family/state/context, retained/hot-request and failure/
all-resource recovery acceptance. Only then speed trials. CPU checks do not prove
physical shared/MTP queue reuse, GPU arithmetic, real memory admission or recovery.
The CLI remains compiled-public/one-request; private caller remains independently
gated. No API/protocol/configuration/Vashti consumer or frontend change; required
Vashti changes: none. Goal active CPU-only. No GPU/model/inference/service/install/
release action, no root operation beyond authorized sudo-cache refresh. Canonical
checkout and inherited untracked cache preserved; no new recovery/throughput claim.

## Cold target scope and shared/MTP source audit (2026-10-17)

Runtime source `811968c38a272916e0bf41620473eb16185dae81` after `46e695a7`, same isolated
branch. This corrects a concrete source integration error found by tracing actual
callers, not by a throughput run. Preserve all prior coupled math and queues.

### Actual cold and captured target ownership

`strata_global_controller` calls `prefill_first_native_window` with the real final
prompt input. `resident_model_batch::prefill_first_native_window` validates the
original Prefill authority, opens the actual backend I/O reuse nonce, enters
`InitialWindow`, and runs `Purpose::AllRowsWithCarry`. Its original family/fault/
I/O closing audit precedes `end_initial`. It grants no long-generation cursor or
closed-request/reset credit. Source Authority tests already cover prompt1 and
longer prompts, initial-only execution and refusal to rearm/close from that state.

The actual native target branch predicate is `purpose==AllRowsWithCarry &&
start!=0 && request_io_reuse_active`. Thus **a one-token prompt starts at0 and
has no named recording**; longer initial prompts can have a capture key. Ordinary
uncaptured LastLogits calls also deliberately have no target capture key. The
previous shared Stream guard incorrectly required Some(key), so it would reject
a legitimate scoped uncaptured target before shared work. CPU compilation/full
shader acceptance did not prove that caller compatibility.

The connected preflight now separates:

- Stream `private_io_active` and `native_submission_scope` establish the actual
  protected runtime/request nonce. These are unchanged, not caller declarations.
- `strata_schedule::shared_recording` accepts the absence of a capture key only
  with a nonzero original request. If a key exists, its nonce must match and its
  original branch must be target, never MTP or accepted-prefix commit.
- Graph `shared_queue` independently checks its actual compute lane0 and absence
  of an active original MTP draft scope. No unkeyed draft-lane exception exists.
  Existing Begin/End nesting, matched original request, MAIN release, node tags,
  real byte hazards, all-peer final fence and terminal cleanup remain unchanged.

These preflights create no scope, source lease, readiness or GPU completion. They
allow actual cold target Nodes to use the same shared schedule as captured ones.
The source comments use prompt-final uncaptured work as the example; the branch
predicate above is the precise condition, not a claim that all initial prompts
are uncaptured. Existing shader math/geometry/private allocations do not change.

### Source review of target/shared-to-MTP continuity

| Requirement | Actual connected source evidence | Remaining device evidence |
| --- | --- | --- |
| Whole target recording capacity | `main.rs` runtime constructor and `daslab_language_probe.rs` shared-client construction explicitly select32768 tasks for Intel core, versus ordinary256. Backend Graph/Planner bound32768 actual nodes. This avoids importing the ordinary small-task automatic-flush policy into the selected source configuration. Arbitrary future private callers must select the same admitted runtime. | Observe real cold/named capture segments and direct hot replay; source limits alone do not prove actual graph fits or is single-segment. |
| Independent shared math and eventual target join | Selected `run_native_intel` forks after actual router/plan, releases MAIN routed segment before fault fold/combined row. `Commands::submit` waits every peer dependency and last copy/shared before the original target fence. Original resources and actual family pools remain in the Recording. | Real queue submission, numerical result and all-peer final-fence behavior under success/failure. |
| Genuine completion before reusing shared queue for MTP | `Graph::complete_lane_submission(0)` waits the exact target flight's fence before `Book::finish_joined`; `prepare_draft_scope` refuses pending/unsubmitted target work and pins the actual initialized full-R source. Existing compute queue1 then hosts MTP. No book bool is used instead of that fence. | Actual shared target completion and subsequent MTP execution, including unknown/partial submission quarantine. |
| MTP begins beside accepted-prefix commit with original R | Original `prepare_draft_scope` submits real MAIN->draft readiness before commit, retains managed R/pin, and scopes all draft accesses. Controller Pending observer borrows original verified R and calls `prepare_next_draft`, preserving full accepted inputs and delivery clipping independently. | End-to-end first/later drafts, rejection/commit, correct carries and resource lifetime. |
| Replay and I/O cannot drop shared owners | Original plan inventory includes every shared descriptor; native I/O reuse excludes named/future/live recording roots and both INFO transfer owners. Existing scoped MTP join leaves MAIN commit debt retained. Shared tag reset occurs only after actual global drain/cleanup. | Retained request reuse, private erasure and full runtime HOST/MAIN accounting/recovery. |

This is a targeted source audit, not completion of the full checklist. Other
artifact/quantized math/prompt/sampling/state/substitution rows still need their
final source review and full protected execution acceptance. No source/placement
or numerical gap is declared solved merely because a helper compiles.

### Validation and current artifact

Receipts under `/media/jmicko/faststore/kuno-development`:

- `strata-shared-cold-scope-final-test-build-20261017.log`, retained runner
  `strata-shared-cold-scope-validate-20261017.py`, and
  `strata-shared-cold-scope-full-cpu-20261017.log`: **559 distinct CPU cases/73
  nonempty suites, zero failed**, retaining all558 prior cases. Added actual
  backend-preflight case checks uncaptured target/nonzero nonce, every named
  target width1..8/state role0..3, foreign/zero nonce, MTP/commit branches,
  wrong compute lanes and original active-draft exclusion. This is policy/state
  source evidence, not actual GPU or runtime request acceptance.
- `strata-shared-cold-scope-{core,off}-clippy-20261017.log`: strict locked release
  **binary** lint passes; changed formatting and whitespace pass. No throughput
  comparison, shader/body rollback, resource-accounting relaxation or new option.
- Normal stable-source builder output `strata-shared-cold-scope-ordinary-20261017.out`
  and directory `strata-shared-cold-scope-ordinary-20261017/` freeze source
  `811968c38a272916e0bf41620473eb16185dae81` with SHA256
  `cd6ee7aca6300e4096bf03bd4433ce03df022a438f8160a99eca289a671217ed`. Independent source/ordinary/frozen hash,
  mode0500 and default Intel features verified. **Frozen binary not executed.**
  A checked source diff confirms every shader producer/body is unchanged from
  `46e695a7`, whose365/139/32 offline receipt remains the shader evidence; no
  unnecessary shader rerun. Do not use the older frozen artifact to test this
  corrected cold-scope admission.

Goal active CPU-only; speed gate closed. No GPU/adapter/model/inference/service/
install/release action, no privileged operation beyond authorized sudo refresh.
No API/protocol/configuration/consumer/frontend change; Vashti changes none.
Canonical/inherited files preserved. No physical correctness, memory recovery,
privacy-forensics or new performance claim. Full rewrite remains incomplete.


## Original artifact and runtime Q8 conversion audit (2026-10-17)

Runtime source `71dceafcd6727bf04cfc85cc6bb3797d08351b97`, pinned MIT upstream
`fb58e0dbc8399662c0e47c76578c6e878b14f6cf`. This is a source/artifact and CPU
conversion check, not device inference or speed acceptance.

The actual two original IQ3_S shards bind through `GgufFile::open_shards` and
`Qwen4Exp::bind`:83,617,662,656 bytes,1,224 tensors,48 layers,512 experts/top10,
640 FFN. Float types remain484 BF16/292 F32/one F16. All48 scalar shared gates
are original BF16 rank-one[2560]. The actual tokenizer has248,320 vocabulary
entries,247,587 merges, qwen35 pretokenizer and EOS248046. Thirty fixed public
UTF-8/ASCII-control/combining/CJK/emoji/long-piece/chat-control cases in BOTH
parse-special modes match pinned Python exact token IDs and roundtrip. Raw IDs
and already-rendered text are the Rust engine boundary: no implicit BOS/EOS or
chat-template rewriting. Upstream `sycl/src/program/generate.cpp` also consumes
pretokenized IDs; `serve/server.py` renders its Jinja outside the inference engine.
This does not assert arbitrary chat-API/template support in Kuno's private caller.

Original `mtp-q2_0.gguf`:889,017,216 bytes/31 tensors binds through actual
`MtpGraph::bind_strata_q2`, with original target binding identity. Both embedding
and Q6 head views genuinely belong to the target, not the sidecar. All nine F32
norm vectors match exactly one `1+w` fold. Header/source identity checks passed;
full-file SHA authentication was not repeated and remains the protected caller's
separate responsibility.

Do not misread the reference-graph comment rejecting converted runtime files as
missing native conversion. Actual `resident_mtp::load_page` already creates Q8_0
in RAM at startup for the nine used large BF16 projections; GR/router/scalar
roles retain BF16 and head remains original Q6. The publisher packs ten matrices,
but its indexer projection/two norms are unused by dense MTP GQA. No runtime
`.bin` model file or second persistent model copy is introduced.

The real correction replaces per-value division/round with `strata_weight_q8`:
F32 maximum/127 -> reciprocal -> multiply -> sign*floor(abs+0.5), plus original
F16 scale storage. F32 addition before floor differs even from ordinary round
immediately below0.5. Synthetic golden controls distinguish BOTH prior division
and reciprocal-with-round from the pinned NumPy packer. Invalid shapes/nonfinite
scales/codes reject before upload. This is public weight conversion, distinct
from native activation quantizers' own rounding contract.

Read-only CPU audit of every converted row/block covers24,832 rows/2,119,680
blocks/67,829,760 codes. All nine complete packed matrix SHA256 streams match
independent pinned NumPy conversion of the original BF16 sidecar;24,743 actual
weight codes differ from the old division path. Eighteen sampled original blocks
and ten synthetic controls also match. Converted weights are hashed in bounded
RAM chunks, never written as model artifacts. These checks prove packed source
bytes, not downstream GPU arithmetic or output correctness.

Receipts under `/media/jmicko/faststore/kuno-development`:

- `strata-artifact-audit-20261017/{src/main.rs,compare.py,inspection-final-packed.json,whole-matrix-final.log,summary.json}`:
  actual artifacts/full nine matrix comparison. `src/bin/tokenizer.rs`,
  `compare-tokenizer.py`, `tokenizer-{rust,summary}.json` retain actual-vocabulary
  controls. Python regex dependency is isolated in that audit's `.venv`.
  Initial one-shard refusal and missing dependency observations are retained;
  final comparison uses both original shards and passes30 cases.
- `strata-weight-artifact-validate-20261017.py`,
  `strata-weight-artifact-final-test-build-20261017.log`,
  `strata-weight-artifact-full-cpu-20261017.log`:561 distinct cases/74 nonempty
  suites, zero failures. Both rounding boundaries plus malformed/nonfinite/zero/
  underflow controls are included. Final logs follow the F32 add-before-floor fix;
  earlier `*-pre-floor.log` are historical.
- `strata-weight-artifact-{core,off}-clippy-20261017.log`: strict binary Clippy
  passes. Changed formatting/whitespace pass. No GPU initialization or inference.
- `strata-weight-artifact-ordinary-20261017.out` and matching directory: normal
  stable source71dceafc, frozen SHA256
  `0193577d5af95f72e213527adc9124f20c28fa309c44ace70bcf0e874679b187`.
  Source/original/frozen hashes, mode0500 and features independently verified;
  executable not run. Shader producers unchanged; source46e695a7's365/139/32
  offline evidence is retained, not rerun for a CPU weight-loader correction.

Remaining concrete source gap: pinned `sycl/src/core/mtp.cpp::record_rest` defaults
`STRATA_MTP_SHARED_BRANCH` on for Intel (non-HIP); it forks shared projection/
scaling beside routed experts and joins before native vec4/FMA combine. Actual
Rust `ResidentMtp::native_moe` still enqueues shared after all routed work and
combines inline with scalar sigmoid. Port these coupled dependencies next: real
scope-aware shared queue, independent fault, original rounded scale, native
combined reduction, all-peer fence and exact owner/pricing/erasure. Do not weaken
existing target/MTP request-nonce/branch or verified-R guards to permit the new
path. Reuse of the two adopted compute queues must be an explicit backend
substitution with actual command-family and pending-commit ownership proof.
Full rewrite and numerical/device/recovery acceptance remain incomplete.


## Connected MTP shared fork and native combination (2026-10-17)

Runtime source `35f0990fe3d89635797870d477ab80e321d47a31`, pinned MIT upstream
`fb58e0dbc8399662c0e47c76578c6e878b14f6cf`. This closes the previously documented
serial MTP shared/routed source gap as one connected implementation block.
No incidental timing or speed result selected this implementation.

Pinned `sycl/src/core/mtp.cpp::record_rest` defaults the shared branch on for
Intel/non-HIP. Its shared projections/scalar/scale run beside router/routed work,
then `native_moe_combine` selects128-thread aligned vec4 output: first rank's
product rounded, ordered FMA ranks1..9, then add the separately stored scaled
shared row. The Rust actual `ResidentMtp::native_moe` now enqueues that order:

1. Real shared queue Begin; original shared gate/up read one native Q8_1 input;
   native fused SwiGLU writes the actual down Q8 planes; original down and BF16
   scalar projections finish; new128-thread vec4 scale stores the rounded row;
   End only restores enqueue tagging, never GPU-completion credit.
2. Owning draft queue performs original BF16 router/top10, S2 prepare, all four
   authentic Q2 GU shards, activate/quantize and all four DOWN shards. Expert IDs,
   source pages, F32 activation scales and original rank destinations stay intact.
3. Explicit routed-segment release before actual shared-fault fold and native
   vec4/FMA combine. Both shared fault and shared row have genuine byte hazards;
   neither side aliases the routed fault during independent work. Existing GR/
   head/sampler/carry consumers receive this combined row.

### Actual scopes, queue substitution and lifetime

`Stream::native_strata_shared_boundary` obtains the existing real private I/O
nonce and optional named inference branch. Commit branches and foreign/zero
nonces reject. `Graph::shared_boundary` independently requires either actual
MAIN/no active draft, or actual draft lane1 with the SAME original active draft
request. The latter scope comes from the existing verified-R pin and real
readiness submission; the shared API cannot create it. Original branch selection
still rejects named target inside active MTP. Uncaptured work gets no invented
capture key. Direct named MTP probes require the same genuine scope; no bypass.

Actual Graph access admission, shared command pool family, submission queues and
timestamp-family selection all call the same checked `shared_peer`: target0->1,
draft1->0. Planner retains three relative timelines (own/copy/shared); draft copies
stay inline on its own compute queue, and a tagged side fill/copy stays on the
opposite compute queue. All original RAW/WAR/WAW, INFO-header dependencies, byte
inventory and final all-peer fence remain; recording signatures retain side and
release flags. Original recording/flight ownership retains every actual source
and scratch until its real fence joins, including partial/unknown submission.
No global/shared fault hazard is suppressed to manufacture overlap.

**Explicit Vulkan substitution:** reuse the two already adopted compute queues
rather than add Strata's separate logical side queue. Target's shared side uses
draft compute; MTP's shared side uses MAIN, AFTER previously submitted accepted-
prefix commit on that physical queue. MTP front/router/routed work remains on its
independent draft compute queue and can overlap that commit. The shared side may
wait behind commit; do not claim identical three-way overlap or numerical/driver
acceptance from source. All compute submits remain on the original serialized
engine thread. The existing transfer authority/worker is unchanged. No fourth
hardware queue, fake fence, allocator release or parallel submitter is introduced.

MTP front workspace adds logical word `SHARED_FAULT` as its26th actual owner,
rounded to32,768 bytes. Prepared-front identity includes that owner; changed owner
rejects. Original workspace allocation/handles/requests feed actual admission,
retained rebind, padding erasure and full close. Native32K private plan grows from
62,586,880 to62,619,648 bytes; global/model quotes consume the same real plan.
The fault stays sticky until full request erasure and is folded on the owning
lane before combination. No per-step host read, copied weight file, hidden model
owner or request-persistent text/image/seed/token data is added. Existing bounded
native queue metadata allowance and actual metadata ceiling remain in force;
physical whole-device/HOST pricing acceptance remains required.

Native activation Q8 producers (`strata_dense_prepare.wgsl` and MTP S2/shared)
now implement SYCL half-away rounding with truncation and a fractional comparison.
F32 add0.5-before-truncate incorrectly rounds the representable value immediately
below0.5; this is DISTINCT from the publisher's NumPy weight packer, which really
requires the F32 add-before-floor implemented by source71dceafc. Keep both contracts.
No additional activation/work-weight root or source-file rewrite is introduced.
WGSL exp and Vulkan scheduling remain explicit backend substitutions.

### Acceptance and limits

Receipts under `/media/jmicko/faststore/kuno-development`:

- `strata-mtp-shared-validate-20261017.py`,
  `strata-mtp-shared-final-test-build-20261017.log`,
  `strata-mtp-shared-full-cpu-20261017.log`:565 distinct CPU cases/76 nonempty
  suites, zero failures. Includes real original target/draft scope preflights,
  opposite peer selection, side-copy policy, independent routed segment and
  shared row/fault joins, original byte inventory,26-owner identity, exact native
  private plan, actual new shader bindings/workgroup/FMA, independent canonical
  Q2 integer math and all prior selected561 cases. This is not a full repository
  unit run or hardware execution. Legacy core fixtures outside these selected
  suites are not silently promoted to whole-suite acceptance.
- Initial `*-initial.log` preserves five failed assertions: one new test confused
  the conservative retained write inventory with a read-only node, one old test
  still prohibited MTP side work, and three old quote checks omitted the new32KiB
  owner. Fixed tests preserve actual producer/consumer ownership and check exact
  new geometry; runtime hazards were not relaxed. The runner now explicitly also
  includes MTP MoE and the relevant joined-cache accounting tests.
- `strata-mtp-shared-{core,off}-clippy-20261017.log`: strict binary lint passes;
  changed Rust formatting/whitespace pass.
- `strata-mtp-shared-rounding-20261017.py/.log`:1,546 synthetic F32 expression
  controls match C `roundf`, including both prior erroneous halfway-adjacent
  cases. This validates the source expression, not GPU/compiler numerical output.
- `strata-mtp-shared-ordinary-20261017.out` and corresponding directory: normal
  stable source35f0990f, frozen SHA256
  `384fcc563a8499e326274555f7fd74a94a290b03bb75cd949a46a9f74a6b8aff`. Source/ordinary/frozen hashes,
  mode0500 and default Intel features independently checked. Only the original
  `--validate-port-shaders` entry (before adapter/vendor/runtime construction)
  ran: `strata-mtp-shared-shaders-20261017.log` records366 offline variants/139
  physical-pointer checks/32 matrix configurations, all passing with pinned
  SPIRV-Tools fbe4f3a. No adapter/GPU/model inference or speed test occurred.

The prior original-artifact/tokenizer/once-folded norm/full nine packed Q8 matrix
receipts remain applicable; this block does not modify their source bytes/load
conversion. Full rewrite remains incomplete. Next finish the whole connected
source/math/prompt/sampling/state/placement/queue/pricing/substitution audit,
then protected numerical count1/multirow/context/retained/hot-request and failure/
all-resource recovery acceptance. Source readiness is not an independent device
correctness oracle. CLI remains compiled-public/one-request; a protected private
caller is independently gated. Only after complete audit and execution acceptance
may live whole-engine throughput work begin. Goal active CPU-only; authorized
sudo refresh only. No service/install/release/API/configuration/frontend changes;
required Vashti changes none. Canonical/inherited files preserved.


## Native routed Q8 and fused SwiGLU producers (2026-10-17)

Runtime source `1a034311407a39d06a714090f82e506b5ef5174a`, pinned MIT upstream
`fb58e0dbc8399662c0e47c76578c6e878b14f6cf`. This replaces another legacy dependency
inside the actual connected global Intel decode path; it is not an optional
speed experiment. Preserve the entire coupled target/MTP/shared branch.

### Source comparison and actual connection

Pinned `sycl/src/kernels/cuda/iq_kernels.dp.cpp` supplies `q8_1_store`,
`native_expert_grouped` and default non-v1 `swiglu_q8_1_entries_kernel`. Q8 blocks
use32 lanes, native XOR16/8/4/2/1 sum, clamped finite scale/sum, half-away codes
computed from the unrounded F32 scale, then original F16 d/s storage. Fused
SwiGLU uses `(g/(1+native_exp(-g)))*up` with product contraction explicitly off
before the Q8 sum. The Rust selected path still used a legacy serial input block
loop and stabilized negative SiLU in separate ACT-write/requantization passes.
Those are now replaced together in `strata_global_moe::Workspace::enqueue`:

- `strata_global_experts::prepare_gu` -> original separate GU roles ->
  `prepare_down` (fused) -> original DOWN role. The old global `activate` call
  and serial global GU/DOWN preparation are removed from this selected consumer.
  Historical feature-off/frozen helpers stay separate; no speed-driven fallback.
- `strata_expert_q8` emits both actual producers and the layout constants used
  by real host launch bounds. Each block reads its authentic32 original values,
  subgroup-reduces max/sum, applies native half-away/clamp and writes original
  packed bytes/F16 d/s into the existing full-plane owners. Second packed planes
  remain zero; their former second-scale slots hold native Q8_1 half sums.
  Native IQ consumers read the same first packed words/half-rounded scales.
- Fused DOWN consumes both actual GU stamps, reads gate/up from the real PART
  owner, materializes the F32 product in640-value Workgroup storage, synchronizes,
  packs Q8 and clears dead gate/up slots. The completed-role header is published
  only after all blocks; UP stamp clears before the original DOWN writer sets
  it. Invalid/nonfinite data keeps readiness absent. No synthetic GPU completion.
  Explicit STORAGE barriers order global zeroing/packing and readiness, alongside
  Workgroup barriers for fault initialization/product/reduction storage.
- There is no intermediate ACT device write/read on this selected path. Its
  original admitted ACT owner remains retained and fully erased, as with the
  pinned grouped scratch's unused h view in fused mode. No private allocation,
  model-byte conversion, CPU route readback or new source transfer is added.
  Actual graph bindings carry PART/status read-write and planes write access;
  original byte hazards, request nonce, capture/replay and full cleanup remain.

Original `row_entries` arithmetic audit: IQ2_S/3_XXS/3_S use native signed
integer finish before F32 conversion; IQ4_XS remains scalar per entry (no native
Multi specialization). IQ4_NL DOWN retains two16-value calls per32 values; Q2_0
retains original signed codes/scale. Roles keep8 lanes per row, native XOR4/2/1,
4/2/1 entry chunks where supported, original top10 destinations and actual
registered VRAM/HOST source pointers. This source review is not device parity.

Independent pinned GGUF header reader verifies all144 installed role tensors:
48 layers,512 experts, GU dimensions[2560,640,512], DOWN[640,2560,512], paired
original gate/up types. GU layers:17 IQ3_XXS/20 IQ2_S/10 IQ3_S/one IQ4_XS;
DOWN:39 IQ4_NL/nine Q2_0. Every format has the selected connected consumer.
No tensor payload/full-file hash is read/repeated by this header inventory.

### Reviewed backend substitutions

The Rust private ABI has padded per-frame headers/readiness and full-owner
inactive erasure. A256-thread workgroup owns one active token (GU) or rank row
(DOWN), iterating independent32-value blocks. Source uses flat block launches;
this changes concurrency within the producer, not a block's values, arithmetic,
XOR order, scale, codes or subsequent consumer. Individual frames are disjoint;
one original frame owner publishes readiness after all its blocks. Frame0 clears
only inactive/header/tail spans that cannot alias another active frame. Actual
host launch and emitted constants share checked Layout bounds; source tests
prove disjoint whole-owner coverage/every column once for counts1..8. This is the
source-level justification for the adaptation, not identical launch/performance
or physical execution proof. No cross-workgroup counter or extra global scratch
is substituted for real completion. WGSL exp remains distinct from SYCL native
exp; numerical/device acceptance is still required.

The required separately rounded F32 product is also enforced in emitted native
SPIR-V: `preserve_native_product_rounding` adds NoContraction only to FMul result
IDs, preserving all instructions, explicit FMA, descriptor interfaces and the
complete original header/bound. The idempotent postpass is consumed by actual
`Graph::register_wgsl_module` after checked HAL Workgroup layout normalization,
and by offline validation. Markers select this routed fused producer, target
shared fused preparation and both MTP fused quantizers; unrelated arithmetic is
unchanged. This is shared producer correctness, not a new math mode or API.
The in-memory program owns the transformed words; no request data enters code.

Native cache source audit found no evidence that old persistent custom WGSL was
masking changes: compilation.cache defaults None and disk insertion requires
an actual SpirV representation, while these custom WGSL tasks return repr None.
This is source evidence, not inspection of arbitrary future caller configuration.

### Validation and next gate

Receipts under `/media/jmicko/faststore/kuno-development`:

- `strata-expert-q8-validate-20261017.py`,
  `strata-expert-q8-final-test-build-20261017.log`,
  `strata-expert-q8-full-cpu-20261017.log`:568 distinct CPU cases/77 nonempty
  suites, zero failures. Includes all565 selected prior cases and actual layout/
  native producer binding/workgroup/storage barriers, every frame/column and
  annotation-only product postpass tests. No full repository/device claim.
- `strata-expert-q8-{core,off}-clippy-20261017.log`: strict binary lint passes;
  changed Rust formatting/whitespace pass. Earlier `*-pre-product.log` are
  preliminary; `*-initial-product.log` preserves an incorrect test assumption
  that NoContraction OpDecorate had4 words (it has3). `*-initial-header.log`
  exposed rspirv assembly's generator stamp; the final pass restores the full
  original header and tests unchanged instructions/explicit FMA/idempotence and
  malformed-module rejection. No failure was hidden or used to roll back a port.
- `strata-expert-format-audit-20261017.py/.json`: fixed public header inventory
  through the independently pinned Python GGUF reader,144 tensors/all48 layers.
- `strata-expert-q8-ordinary-20261017.out` and matching directory: normal stable
  source1a034311, frozen SHA256 `d3473d8d509d1ed8e92f98192a7a258160a0008c30f8b0bd01b4c0e2d999a25f`;
  source/ordinary/frozen hashes, mode0500/default Intel features independently
  verified. Only original pre-adapter `--validate-port-shaders` entry ran.
  `strata-expert-q8-shaders-20261017.log`:365 variants/139 physical-pointer
  modules/4 rounded-product modules/32 matrix configurations pass with pinned
  SPIRV-Tools fbe4f3a. New product modules use the same HAL layout normalization
  and NoContraction pass as actual native registration, not raw WGSL alone.
  Global compiled family now has13 variants instead of14 because a separate
  activation dispatch disappeared; this is not omitted model format coverage.

No GPU/adapter/model inference, throughput ranking, service/install/release or
API/configuration/frontend action occurred. Sudo refresh only. Original artifact/
tokenizer/full nine packed MTP-weight receipts remain applicable. Full rewrite
is incomplete: finish the remaining connected source/math/prompt/sampling/state/
placement/queue/pricing/substitution audit, then protected whole-model count1/
multirow/context/retained/hot-request and failure/all-resource recovery before
speed. CLI stays compiled-public/one-request; private caller independently gated.
Goal active CPU-only. Required Vashti changes none; canonical/inherited preserved.


## Native tiled accepted-prefix GDN commit (2026-10-17)

Runtime source `39c8d9e1b37700c0adeab0556e2d5c2123dbaffd`, pinned MIT Intel upstream
`fb58e0dbc8399662c0e47c76578c6e878b14f6cf`. This source audit found and replaced a
missing default kernel, without using speed results to select an implementation.
The whole rewrite remains incomplete and its speed gate stays closed.

### Pinned source and connected consumers

`sycl/src/kernels/cuda/verify_kernels.dp.cpp:218..300` defines
`gdn_step_commit_kernel`. The actual `gdn_step_norm_multi` wrapper at 1933..1965
defaults `STRATA_GDN_COMMIT_SPLIT` on; when n_keep is present and no output rows
are requested, it selects this kernel before the generic 512-thread variant.
Optional `STRATA_GDN_SPLIT` remains off for the selected path; it is a different
mode, not the prerequisite implemented here. `sycl/src/core/verify.cpp:1710+`
records the real retained commit, and 2371+ skips it only for count 1 self-commit.

The actual Rust `strata_gdn_replay::enqueue_prefix` now selects a dedicated
`strata_gdn::Kernel::Replay` source with 128 threads and grid[value_heads,4,1].
Both original native pending-prefix commit and older diagnostic repair consume
it. Host launch rejects any different Replay grid. One workgroup owns 32 state
columns; four groups cover the 128 columns for each value head. Each thread owns
32 original rows in one of four row groups, using native row/value-head-major
address `((row*HV+head)*128+column)`. All 128 threads jointly stage a full key head.
Each accepted token computes 32 ordered FMA partials, sums groups 0/1/2/3 in order,
forms native delta and updates 32 rows with explicit FMA. This does no query,
readout, Z/gamma, output RMS or output tensor work.

Only seven actual storage bindings are passed: retained normalized key, value,
alpha/beta controls, prewindow recurrent state, paired candidate state and two
original status owners. Query/Z/gamma/GATED are no longer passed to Replay.
Original InputCache allocation/retention/erasure remains intact for the other
real consumers; no ownership is fabricated from a matching shape or count.
Actual INFO AcceptedCount/Position, source stamp, original pending receipt,
public origin, per-family permits, full fault join and native role/cursor commit
remain mandatory. Cold/warm commit recording still consumes genuine runtime
accepted count. Count1 still uses its original candidate; multirow all-accept
still commits from retained original inputs. Convolution/history and downstream
fault fold retain their previous actual paths.

Workgroup barriers protect key staging, partial reductions and the local finite
fault before final stores. Unique tiles write disjoint state/status cells.
Nonfinite data clears affected tile readiness and writes zero candidates; the
existing full-state/status fault consumer rejects the entire sequence. A tile
cannot grant another tile or family completion. No GPU owner, admission quote,
model conversion, source transfer, queue or request-content persistence is added.

### Explicit backend substitutions and remaining gap

Original Strata commits its state in place. The existing Rust protocol reads
an authenticated prewindow OLD owner and writes its distinct uncommitted NEXT,
then swaps roles only after real joint acceptance. This rollback/ownership
substitution remains explicit and requires device/context/failure acceptance.
Keys/values are separate original cached arrays rather than one contiguous hbuf;
controls remain one original gate/beta owner. Native arithmetic/column grouping
is preserved. Vulkan exp/FMA/device compilation remains numerically unproven.

A further concrete source/data-movement gap remains: native verification's
`ALL_OUT` 512-thread kernel keeps speculative state in registers and does not
store the original state (`verify_kernels.dp.cpp:484..487`); current Rust Step
still writes a provisional NEXT at every multirow verification. Ordinary prompt/
full-row calls also use Step and need their candidate. Resolve actual verification
intent and capture-key/caller/acceptance dependencies together before suppressing
these stores. Do not infer authority merely from count>1 or conflate it with the
new accepted-prefix kernel. The dedicated commit here does not close that gap.

### CPU-only evidence and limits

Logs and reproducible runner under `/media/jmicko/faststore/kuno-development`:

- `strata-gdn-commit-final-test-build-20261017.log`: locked selected release test
  build. `strata-gdn-commit-full-cpu-20261017.log` ends 572 distinct passes across
  78 nonempty suites. The original 77-suite run covers 568; the appended real GDN
  suite covers four additional cases, independently deduplicated from executable
  test listings. `strata-gdn-commit-numeric-cpu-20261017.log` retains those four.
  Runner `strata-gdn-commit-validate-20261017.py` now includes all 78 filters.
- Two new cases cover every actual tile/state cell once for all 1..48 heads and
  every 1..8 accepted prefix against independent F64 full-matrix recurrence for
  multiple value heads sharing keys. Existing native AB and recurrence oracle
  controls pass too. These CPU models do not execute WGSL on a physical device.
- `strata-gdn-commit-{core,off}-clippy-20261017.log`: strict locked release binary
  lint passes in both configurations. Changed Rust formatting and whitespace
  pass. This is selected coverage, not the entire repository test suite.
- `strata-gdn-commit-ordinary-20261017.out` and its ordinary build directory:
  stable-source frozen artifact above. Independent HEAD/source-file/JSON,
  ordinary/frozen hash, mode 0500 and exact Intel features verified.
- `strata-gdn-commit-shaders-20261017.log`: only that artifact's early
  `--validate-port-shaders` branch ran, before vendor/adapter/device construction.
  All 365 variants/139 physical-pointer checks/four product modules/32 matrix
  configurations pass using pinned SPIRV-Tools fbe4f3a, not the older system tool.

No new GPU/model inference, numerical device/whole-model, physical recovery,
throughput or deployment claim. No API/configuration/consumer/frontend changes;
required Vashti changes:none. Canonical checkout/inherited cache preserved.
Goal stays active CPU-only; sudo refresh was authorized and observed successful.
Complete the listed source gap and remaining whole-branch audit, then protected
execution acceptance before any live speed trial.


## Deferred verification state and history-only GDN commit (2026-10-17)

Runtime source `7a0489d38bdce98961901759cbe65d95a1db33ba`, pinned MIT Intel upstream
`fb58e0dbc8399662c0e47c76578c6e878b14f6cf`. This closes the concrete GDN source gap
listed in the preceding receipt and connects its coupled history/recording/commit
dependencies. Preserve the full branch; no speed result selected this change.

### Actual source and data flow

Pinned `sycl/src/core/verify.cpp:747,964..986` selects self-commit for the ordinary
one-token path and passes no n_keep for multirow verification. Actual
`verify_kernels.dp.cpp:484..487` writes recurrent state only for committing calls;
ALL_OUT verification keeps it in registers. `gdn_conv_l2_multi_kernel:44..99`
writes history only when its commit argument is true. `capture_commit:1710+`
uses `gdn_conv_commit` plus the state-only recurrence; the 256-thread history
wrapper is at 1686..1706, and its kernel at 102..120. No convolution or normalized
Q/K/V recomputation is part of that commit.

The connected Rust path now follows those dependencies together:

1. `resident_model::batch::Workspace::run_global` derives verification intent
   from its actual scoped PrefixSelection consumer. `prepare_batch` stamps
   `Metadata.defer_state = verification && count>1` before the original stage
   anchor is made. The native/prompt guards and original checked 1..8 extents
   remain. Legacy ordinary enqueue supplies false. Count alone cannot authorize
   deferral; count-one still has its self-commit candidates.
2. Actual GDN enqueue selects `Kernel::ConvVerify` and `Kernel::Verify` only for
   that authenticated mode. These are source-specialized STORE_HISTORY=false
   and STORE_STATE=false variants. They retain original normalized Q/K/V,
   controls, recurrence/readout/norm/output and all local finite checks, but do
   not store provisional NEXT history/state. Ordinary Conv/Step and the separate
   large prompt GDN path retain their real state/history writes.
3. `NativeStrataBranch` includes one static content-free state-mode bool in its
   derived equality/hash, with checked target-only setter (deferred count-one
   is rejected). Default constructors stay false. `run_global` records its real
   mode. Both full-row and verification calls retain original capture/replay;
   they cannot select each other's command inventory despite identical row,
   context, roles, request nonce or model owner. No invocation/position/token
   is inferred from matching parameter bytes. No dynamic GPU INFO field added.
4. `strata_target_replay::Completed` carries the mode from the exact key whose
   genuine runtime submission completed; GDN compares it with the original
   receipt mode along with its existing original fault/position/count/role/
   source-owner checks. Warm replay still retains genuine inputs and does not
   manufacture acceptance or current state from the key.
5. Actual native `strata_gdn_replay::enqueue_prefix` now enqueues
   `Kernel::ConvCommit` with grid[ceil(C/256),1,1], followed by the previously
   connected tiled 128-thread recurrence and original complete fault fold.
   ConvCommit uses original retained RAW and prewindow history to write exactly
   the last three RAW rows for accepted n; it binds no convolution weights or
   Q/K/V outputs and repeats no SiLU/L2 math. Real AcceptedCount and Position
   schema, original capture identity, retained roots and caller proof remain.
   Diagnostic repair uses this same math with its original opposite root roles
   and constant accepted count. Count-one native commit still skips the graph.
6. The original pending-prefix preflight requires the mode expected of the actual
   verified-prefix transaction. Before any actual role swap,
   `candidate_commit_extent` refuses deferred receipts without a selected prefix,
   including all-accept; original join/fault/all-family permits still establish
   device acceptance. Wrong mode fails original anchor equality. No bool alone
   proves GPU completion. Failure remains terminal through existing cleanup.

### Substitutions, bounds and persistence

History/state publication still uses the original Rust read-only OLD and paired
uncommitted NEXT followed by actual accepted role swap, versus upstream in-place
commit. This already admitted rollback/ownership substitution is preserved.
Cached normalized K/V are separate original owners rather than contiguous hbuf.
History commit additionally checks finite retained RAW/history and writes both
original readiness owners for the accepted span, supporting the existing full
fault consumer; source native history commit has no such explicit fault outputs.
It computes only history selection and these finite/readiness checks.

New modes add no GPU root, scratch allocation, state copy, model-byte conversion,
transfer, hardware queue or GPU INFO field. All old retained buffers/padding are
still charged and erased. Static host key/receipt/completion metadata gains its
mode bit; receipt and completion fields are zeroized. Runtime parameter/graph
metadata accounting uses actual `size_of` and capacities, so no fabricated host
credit is taken. The selected target/commit/MTP shape inventory remains 122 keys
within the original 128-branch limit. An optional mixture of additional full-row
and verification recordings still shares that existing bounded cache; this is
not a claim of unbounded arbitrary-mode/context residency. Native math/exp,
physical graph replay/state publication and recovery remain unverified.

Kuno still persists no live prompt, output, request seed/token, state image or
content hash. This work writes public shader sources, synthetic numeric tests,
fixed metadata diagnostics and build logs only. No private inference was run.

### Exact CPU-only validation and limitations

Evidence under `/media/jmicko/faststore/kuno-development`:

- `strata-gdn-deferred-final-test-build-20261017.log` and
  `strata-gdn-deferred-full-cpu-20261017.log`: locked release build and 576 distinct
  selected cases/78 nonempty suites pass. This is not every repository test.
  New coverage: real parameter-program cache misses in BOTH cross-mode
  directions for every multirow count/role; exact selected 122-key bound; original
  anchor rejecting changed mode and zeroizing it; every accepted-prefix extent
  refusing ordinary publication of deferred candidates; independent streamed RAW
  history oracle with exact tile/channel coverage, including partial 256-lane tails.
  The namespace case runs through both existing vendor-module include sites
  (two distinct executable case names, one test body); counts do not imply 576
  independent mechanisms. Prior F64 recurrence/state tiling/AB cases pass unchanged.
- `strata-gdn-deferred-{core,off}-clippy-20261017.log`: strict locked release
  binary lint passes; changed-source rustfmt and git whitespace checks pass.
- `strata-gdn-deferred-ordinary-20261017.out` and its ordinary directory: stable
  committed source, exact default Intel features, source JSON/file/HEAD, ordinary
  and mode 0500 frozen SHA256 independently agree (header above).
- `strata-gdn-deferred-shaders-20261017.log`: only that frozen executable's
  pre-adapter `--validate-port-shaders` ran; 368 variants/139 physical-pointer
  checks/four product modules/32 matrix configurations pass using pinned
  SPIRV-Tools fbe4f3a. Three additional GDN variants account for 365->368.
- Preliminary `*-initial.log/.out` records 574 passes for the intermediate
  nonfinal mode/capture design. It is not final acceptance. The subsequent
  `*-missing-fixtures.log/.out` test build failed because a text edit removed
  original quote helper fixtures; they were restored verbatim from original
  HEAD. A new exported method also received its required documentation before
  the final rebuild. Final source retains BOTH capture paths and all original
  quote fixtures; neither their assertions nor the resource budget was relaxed.

No physical GPU/model/whole-inference, numerical/device context, hot-request,
recovery, throughput, install or release claim. Required Vashti changes:none;
public API/configuration unchanged. Canonical/inherited work is preserved.
Whole rewrite incomplete, goal active CPU-only. Next QSA source/state audit and
remaining full-branch math/prompt/sampling/placement/queue/pricing/substitution
checks, then protected whole-model correctness and all-resource recovery before
speed. Sudo refresh only was authorized; it is not permission to use the GPUs.


## Native QSA window indexer and direct pooled scoring (2026-10-17)

Runtime source `e5981a138a58a4209448d8091b1f576c3b1968b4`, pinned MIT Intel upstream
`fb58e0dbc8399662c0e47c76578c6e878b14f6cf`. This is a connected source rewrite,
not a speed experiment. Full rewrite remains incomplete; speed gate stays closed.

### Source finding and actual selected path

`sycl/src/core/verify.cpp:1087+` calls
`native_qsa_indexer_append_steps` for the selected window. Actual
`sycl/src/kernels/cuda/native_qsa_indexer.dp.cpp:376..465` dispatches ONE 256-thread
workgroup and its `append<false>` processes all rows in order. The separate C-2
`append_batch`/`append_blocks` is another producer; do not substitute it for this
selected decoder. `append:53..179` rounds incoming keys through F16, updates the
three-row tail, and only for position 0 or a completed four-row block computes
mean, two-level subgroup32 RMS/gamma and 64-dimension rotation. It retains pooled
keys and the fixed first-cell dead/spare key. QSA scores consume retained pooled
values with causal completed-block bounds, using the spare for a partial block.

Prior Rust decode copied every retained pooled prefix into per-query POOL,
computed new means in a 64-thread kernel, dispatched separate norm/RoPE for the
new blocks and republished those keys. The new connected path is:

1. Actual `resident_qsa::batch::enqueue_native_qsa_operation(Operation::Pool)`
   calls `Kernel::Indexer` with one 256-thread workgroup over the real CACHE,
   CAND, original index norm gamma, persistent POOLED and original fault owners.
   `strata_qsa_indexer.wgsl` stages at most three required pre-window RAW/F16
   index keys, consumes each original packed candidate row, updates its local
   tail and writes only newly completed pooled keys (plus first-cell spare).
   Each completed block keeps chronological F32 sums, native SCALE FMA0.25,
   XOR16/8/4/2/1 warp sums, eight-part second warp reduction, shared normalized
   values and original block-position rotation. Actual ordinary registration
   and offline validation apply the existing product-rounding postpass here;
   explicit FMA remains unchanged. Actual native WGSL registration already
   requires subgroup32 (`strata_graph.rs::register_wgsl_module`); wrong subgroup
   additionally faults rather than inventing readiness.
2. `Kernel::IndexerFlags` runs after that actual producer and mirrors its genuine
   global fault into the original scoped preparation/normalization flags for
   each query's completed blocks. Original queue dependency analysis sees the
   real POOLED/fault writes and flag reads; all original status/fault joins and
   family acceptance remain required. The compatibility flags alone are not
   completion or numerical authority; later original pack/other status folds
   still invalidate a window if their source is bad.
3. Decode's PoolNorm is now a no-op because the actual producer already computed
   the native normalized/rotated keys. `Kernel::DirectScore` consumes the actual
   persistent POOLED root, never per-query copies. It uses only blocks completed
   at that query's own position and the separate retained spare for its partial
   tail, even when a later row has already completed that same block. Original
   dot/reduction/top-k/attention paths and source candidate/cache ownership stay.
4. Native decode skips MEAN/POOL clears in both ordinary and grouped-clear
   branches. Those two original full owners remain in layout/admission/retention/
   rebind/full-request erase; they are not refunded, aliased to untracked memory
   or promoted to public state. Large prompt and feature-off paths keep their
   original producers and clearing. At 32K/MAX8 the two logical views are 32MiB
   each; this removes their per-window traffic, not their conservative admission.
5. Existing accepted-prefix CLEAR/FOLD keeps accepted cache/POS and completed
   pooled blocks, clears rejected rows/newly completed suffix blocks, and leaves
   the immutable spare intact. A continuation reconstructs its required at-most
   three raw keys from that accepted cache. All-accept remains the original
   no-suffix-clear case. No synthetic tail state or alternate expert/router IDs.

Original INT8 KV packing was also compared with actual
`sycl/src/kernels/cuda/kv_q8.dp.cpp:36..93`: original 64-value groups, two warp maxima,
F16 stored scale, codes against THAT rounded scale, nearest-even conversion and
[-127,127] clamp are retained. Rust combines two independent 64-value groups in
one 128-thread owner block and interleaves codes/scales in its existing packed
cache ABI. This is a reviewed launch/storage substitution, not a bit-parity or
physical-placement receipt. No weight/model format or MTP GQA path is changed.

### Reviewed storage/math substitutions and remaining scope

Source uses a separate persistent 3x128 tail plus a prewindow tail snapshot and
reappends accepted original keys after restoring it. Rust's existing packed cache
retains the exact FP16-rounded RAW index keys. The new producer reads only the
at-most three necessary prefix cells from that original GPU owner into Workgroup
storage, versus maintaining a duplicate persistent tail. The original accepted
cache and pooled suffix transactions therefore remain the authoritative rollback
path. Explicit GPU input/output dependencies are preserved. This adds no HOST
content mirror, private GPU root, full-cache copy or transfer.

Source places the spare after the last complete block; Rust already retains it
in the fixed final pooled slot and the scorer explicitly chooses that slot for
incomplete blocks. The new producer preserves that reviewed ABI substitution.
Logical block RoPE positions remain 4*b in the selected linear-position 32K model;
nonidentity RoPE/table modes are not claimed. Vulkan pow/sin/cos/inverseSqrt and
fused rotation still require device numeric acceptance. This block does not
close QSA projection/norm/selection/attention, original host/device KV placement,
physical replay/rollback or whole-branch acceptance by implication.

### CPU-only evidence

Evidence under `/media/jmicko/faststore/kuno-development`:

- `strata-qsa-indexer-final-test-build-20261017.log` and
  `strata-qsa-indexer-full-cpu-20261017.log`: locked selected release build,
  581 distinct selected cases/80 nonempty suites pass. Previous 576 cases remain;
  five added executable cases are the two existing QSA packing/attention oracles
  and three new native-indexer cases. Not the entire repository test suite.
- New tests cover every small prefix/window length 1..8 across all block offsets,
  every accepted prefix followed by a different continuation, exact RAW F16
  chronological means, causal completed/spare choice and rejected pooled suffix.
  Two-level subgroup F32 norm/rotation is compared with independent F64 math,
  explicitly holding the public F32 angle coefficients fixed (not claiming GPU
  transcendentals parity). Original packed owner/tail addresses are bounded at
  both early positions and the 32K boundary. These are CPU mirrors/oracles, not
  physical execution of WGSL or a GPU state-recovery receipt.
- `strata-qsa-indexer-{core,off}-clippy-20261017.log`: strict locked release binary
  lint passes. Changed-source rustfmt and git whitespace checks pass.
- `strata-qsa-indexer-ordinary-20261017.out` and its ordinary build directory:
  stable committed source, exact default Intel features, independently agreeing
  HEAD/source-file/JSON/ordinary/frozen hash, mode 0500. SHA256 is in the header.
- `strata-qsa-indexer-shaders-20261017.log`: only that frozen executable's
  pre-adapter `--validate-port-shaders` entry ran. 371 variants/139 physical-pointer
  modules/five rounded-product modules/32 matrix configurations pass with pinned
  SPIRV-Tools fbe4f3a. Three new QSA variants account for 368->371; original format
  coverage is not dropped. The same HAL Workgroup normalization/product postpass
  as actual runtime registration is used. No GPU/model inference or timing trial.

No API/configuration/consumer/frontend changes; required Vashti changes:none.
No service/install/release action or new persistence path. Canonical checkout
and inherited untracked cache are preserved. Goal remains active CPU-only;
sudo refresh only was authorized. Finish the remaining QSA/full-branch source/
placement/substitution audit, then protected numerical/context/retained/hot-request
and failure/all-resource recovery acceptance before any speed test.
