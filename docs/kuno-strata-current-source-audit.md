# Current Strata source audit

**Current deployed engine checkpoint (2026-10-19):** prompt-optimized worker
60c20e36 / SHA1db4664d is installed through the unchanged image-enabled gateway.
See [current coordination](kuno-vashti-coordination.md),
[service acceptance](kuno-native-text-service.md#native-prompt-processing-optimization-2026-10-19)
and [matched public measurements](kuno-prompt-processing-profile.md).
The following older checkpoints are historical and do not override those records.


**Installed Kuno-only full-toolbox repair (2026-10-19 UTC).**
Worker sourcea6b21a38 / SHA7bc1633b, unchanged image-enabled gateway36626d30 /
SHAd3ed07f4. The stale2048-token PLE history guard now shares the real4096-token
GPU chunk bound; MAX8 decode and all fence/erasure checks remain unchanged.
146 library tests pass,2 ignored;10 PLE regressions and strict relevant Clippy
pass. The previously failing compiled-public8,445-token request now completes
across three chunks; staged10-tool4,207-token streamed call/result, zero writes,
unload, Ollama coexistence and physical recovery pass. The official transaction
installs the exact frozen worker with the complete prior configuration preserved.

Actual live Vashti passes its previously failing prompt with Notes, Calculator
and Images ALL enabled: thinking -> calculator -> result -> final answer. No
frontend/API change. Installed identity/mount, swap/core0, worker write_bytes0,
inspection denied, no OOM/driver faults, services/admission healthy all pass.
Native weights intentionally retained on Intel33,348,456,448B; NVIDIA compute0.
No fresh image/individual-tool suite, new comparable throughput test or release.
Context remains512..32768; above32K and loader/kernel work remain separate.

Read the [active coordination handoff](kuno-vashti-coordination.md) and
[exact service contract and receipts](kuno-native-text-service.md#kuno-toolbox-prefill-repair-2026-10-19)
before continuing. Earlier entries below are historical and superseded only
within the explicitly accepted scope.

**Selected Intel rewrite accepted for fixed-public engine measurements (2026-10-18).**
The connected Rust port is now past the source-wiring, representative numerical,
whole-model completion and physical recovery gates for the selected B70 / original
DASLab IQ3_S / original Q2 MTP configuration. See the
[completion decision](kuno-strata-current-source-audit.md#selected-engine-completeness-and-correctness-decision-2026-10-18)
for exact scope and exclusions. This is not private-caller, service, installation
or release acceptance.

Runtime source `299579b52f9e4b9d8feaaaedb41eec4c193fcb83`, frozen artifact
`strata-fp16-preserve-coupled-20261018`, SHA256
`852a0dbd7dbdd11a3a1eb8e3447a9006dbe7fae2b56d8d3ecd1fb3ccd7c473e1`.
`f0707ddb` requests supported FP16 DenormPreserve16; `299579b5` applies the
same transformation to the actual native replay module as the HAL module.
All74 original-weight operator cases now pass the ORIGINAL IEEE reference and
unchanged tolerances, including FP16/BF16 counts1/8/35. Maximum matrix error is
4.2915344e-6. The old29-mismatch/FTZ diagnosis remains failed historical evidence;
it is not used as the new reference. The aligned embedding fix remains exact.

Protected `strata-fp16-preserve-retained-20261018` passes two requests through
one loaded Session: identical362 outputs/EOS,110 windows and31 real repairs per
reported second request. Protected `strata-fp16-preserve-prose-20261018` passes
512 outputs/no EOS,194 windows and83 repairs. Both display coherent public prose
without saving it; target/draft request erasure, final all-peer close and physical
recovery pass. Final Intel50,688,000B, NVIDIA compute0, all HTTP200, normal admission
and zero classified driver-fault deltas. Swap/core protection was observed active.
633 selected CPU cases/88 suites, core/off strict binary Clippy and pinned offline
397 shader variants/32 matrix configurations pass. Artifacts are under
`/media/jmicko/faststore/kuno-development`.

The speed gate is now open ONLY for the exact accepted artifact and unchanged
fixed-public retained configuration. The committed acceptance manifest is checked
by the outer timing runner. Timing uses trusted static phase lines and writes
only numeric results after successful repeat/erasure/physical recovery; no kernel
profiles, response contents or content hashes. The first accepted trial
`strata-accepted-retained-throughput-20261018` completes successfully:19.7811 and
20.6089 delivered tokens/s for362 outputs on each request, with17.5652s warm
generation and5.2670s warm prefix/preflight. Full warm request before erasure is
22.8322s (15.8548 tokens/s); selected runtime setup is101.1493s. These are
last-prompt-inclusive engine rates, not pure GPU decode. Final physical recovery
again passes with the same clean baseline. The30 tokens/s target is not met.
The original execution-check receipts retain their static `benchmark=false` and
`whole_rewrite_accepted=false` fields; the documented audit and separate timing
receipt provide the later decision without rewriting historical evidence.
No Vashti consumer/API change, installation, reset, reboot or release occurred.

Actual selected-path operator profiling now passes on source`f38629a9` after a
strict rejected first attempt. `strata-selected-operator-coupled-check-20261018`
completes362 outputs/EOS,110 windows/31 repairs, full target/draft erase/close and
normal physical recovery. One serialized target suffix sample totals126.456ms:
Down32.163ms, Gate26.715ms, Up26.589ms (67.59% combined). BF16 matrix work is only
2.996ms. This is operator diagnosis, not a new throughput rate. Baseline stays
19.7811/20.6089 tokens/s on frozen299579b5;30 remains unmet. Next substantive GPU
candidate: remove the selected expert kernels' extra full-window activation SLM
staging and read the original already-quantized input planes directly, as upstream
row-entry dot products do. Preserve 4/2/1 chunks, real source pointers, exact
integer finishing/rounding, reduction order, target verification and all scopes.
See the current source audit's latest operator receipt before changing kernels.

The follow-up counter-only candidate `b7cb8e85`/SHA256
`aa12c4732d21b269beac5745ba876a6df6ca0034607daa9e6a8f93299e00bb9e`
also passes the two-request362-token repeat/erasure/physical recovery check.
It only exposes existing aggregate graph counters at three static public phases;
no inference math, placement, queue work or timestamp-query changes. Warm generation
records15.8413s native fence wait,0.1704s capture,2077 submissions/1964 replays.
This is a diagnostic, not a new throughput rate or replacement for the exact
accepted artifact. Next: profile actual GPU kernels in the selected controller;
CPU command capture is not the leading candidate. Both test units are terminal,
admission/services/memory recovered and shared-shell sudo was refreshed.

**Earlier checkpoint, superseded by the numerical results above:**

Source-audit baseline: `fad14f5dba7395a66cc2c06aea649fe1feeb1ca6`, isolated `codex/strata-rust-port`.
Current inference corrections reach `d271eb492c40023d0008fcdf5bf6219f88a90000`.
Protected execution/target+draft erasure/final recovery now pass, but the public
code output is numerically wrong (special-token loop). Whole rewrite and speed
acceptance remain unproven. Diagnostic72f0b24a adds actual native GR versus CPU
oracle comparison in the fixed public operator probe; `4a712a0e` admits its
explicit compiled inline dispatch. Both diagnostic runs stopped before the
numerical comparison (dispatch guard, then GGUF-open error). Latest recovery
passes with no GPU unit left running. The goal is paused for the human's app
update. See the latest [checkpoint](kuno-native-text-checkpoint.md); resume at
the diagnostic loader, not throughput testing.
Pinned Intel upstream: `fb58e0dbc8399662c0e47c76578c6e878b14f6cf`.
This is the current audit index, alongside the historical evidence in
[kuno-strata-rewrite-completion.md](kuno-strata-rewrite-completion.md).
**Selected engine source-wiring audit passed; whole rewrite acceptance remains
open.** See the explicit decision and scope below. Physical correctness,
resource recovery and performance are separate gates. Human B70 testing/unload authorization is now granted; see the current
protected-device checkpoint for execution findings. This document itself grants
no GPU/service/install/release authorization.

## Entry, loading and cleanup: call chain traced

The selected executable's actual `daslab_language_probe.rs` entry constructs the
owned request and normally calls `strata_engine::execute`; explicit public repeat
validation calls `with_loaded_session`. Both own the same checked loaded Session.
`Session::load/load_inner`
requires the exact Intel device/client, original bound target/draft artifacts,
protected/exclusive caller contract and disabled operator profiling. It creates
Tables, GpuExecutor, the actual admitted target and shared Q2 MTP owner; it does
not delegate inference to Strata or a foreign model engine.

`resident_model_batch::prepare_global_sources_before_prefill` invokes
`executor::prepare_native_static_weights` before expert allocation.
`strata_static_weights::prepare_native_static_weights_inner` validates original
binding, joins each upload, accounts all selected dense tiles, then seals the
inventory. `executor::native_expert_capacity` comes from the shared MAIN budget
minus the fixed dense/private/runtime/public/source/MTP/table inventory.
`strata_global_sources::Sources::load` owns the ranked Cache, the original HOST
complement, address tables and execution queue, with no legacy banks alongside.
The original header/packing/conversion/tokenizer receipts remain necessary;
this call-chain review alone does not reconfirm all model bytes or placement.

`Session::run` consumes original erasure receipts before hot rebinding, invokes
the controller, bounds token delivery and requires actual delivered counts.
Success erases request state in both target and draft while retaining weights.
`Session::close_inner` attempts all peer cleanup paths even after an error,
checks shared-root recovery and retires failed contexts. Source-owner closure
joins readers before HOST detach/free; failed close quarantines roots instead of
pretending completion. **Single-request device completion, no-swap protection, actual target/draft
erasure and final physical recovery now pass atd271eb49. Repeat-request and
numerical correctness remain unverified; generated content is currently wrong.**

## Dense and ranked source placement: actual owners traced further

`executor::begin_resident` constructs actual `strata_static_weights::Plan` and
subtracts its owned bytes, target private state, baseline, I/O/runtime, public
pages, source windows, original MTP capacity and48 source-table roots from the
shared MAIN grant before computing `native_expert_budget`. The complete selected
static matrix inventory owns3,644,840,960B at256-byte alignment, apart from the
separately accounted shared head/vectors. `prepare_native_static_weights_inner`
checks original graph binding and absence of old cache/banks/sources, loads and
joins every exact tile and seals only after actual cache count/bytes equal that
inventory. `resident_model_batch::prepare_global_sources_before_prefill` consumes
that sealed preparation before expert allocation. No old fixed5GiB allowance,
second legacy uniform bank, or resident ghost expert supplies this selected path.

Pinned Intel `generate.cpp` constructs dense/session/draft/head owners before
`ExpertCache::open_sized`. Its native-pack profile walk stops at the first
aligned heterogeneous blob beyond its byte cap; `open_sized` aligns slots256.
Rust `strata_global_plan::Plan::from_graph/from_roles` derives all144 role sizes
and types from the actual original tensors, uses the same complete ranked pair
profile (its full fixture and pinned data both hash
`8f59b4aa8873209dff11c11e37bcda9529a1335b724a1afeea37bf6388975baf`),
stops at that first non-fitting triple and preserves exact256-byte slot
order. The byte cap comes from Kuno's explicit original-owner admission rather
than copying Strata's free-memory heuristic/reserve; physical fit/cache capacity
and backend allocation pricing still require acceptance.

Rust divides native's one contiguous expert arena into <=512MiB owning roots,
without splitting a slot, while HOST complements use <=128MiB role pieces.
`strata_global_gpu::Cache::load` initializes every actual root and all padding
from bounded16MiB original-byte upload windows; each window joins before staging
reuse. No full GPU-root/model host mirror is allocated for startup. `Host::load`
creates the original nonlocal HOST-visible complement and shared target/MTP
embedding pages under the sole immutable registry; its quote includes actual
heap alignment, metadata and bounded setup windows. This is the deliberate
Vulkan allocation/registration substitution for original SYCL/model source
pointers. It preserves actual weights and role offsets, not only residency IDs.

`Sources::build_tables` obtains device addresses only for original complete
registered owning resources, rejects duplicate roots/unaligned/out-of-owner
ranges, and publishes each expert/role's actual address/extent in its original
layer table with retained physical READ dependencies. Target/device routing and
prompt grouping consume that same plan. Missing GPU experts remain original
HOST sources; the default selected NO_HOST path does not invoke CPU expert math
or upload every miss. Original source/binding checks surround publication.

Prompt lending is also connected to actual owners. `prompt_loan_quote` chooses
three real consecutive fitting tail roots, with a bounded original-weight HOST
shadow for only those roots. `begin_prompt_loan` joins prior use, refreshes the
shadow, redirects every lent expert pointer/read dependency to original HOST
bytes and joins publication before registering any lent root as private scratch.
The prompt Scope ends before `finish_prompt_loan`; original private close/erase
receipts are checked before restoring original bytes, source tables and real
queue completion. Partial failure retains source exclusion and retires the
shared context; it cannot publish original model residency. Embedding consumers
must drop their actual owner loans before HOST detach. Physical loan/restore,
source placement/RSS and retained multi-request recovery remain unverified.

## Sampling and accepted-token flow: selected source traced

Pinned `sycl/include/strata/sycl_math.hpp:18..26` selects F32 sampler accumulation
by default; FP64 requires a separate build flag. Do not infer FP64 from the CUDA
sibling or stale comments. This is a selected-source statement, not independent
attestation of all historical build/environment overrides.

The actual default split path in Intel `sampler.dp.cpp:1566..1615` matches the
Rust `strata_sampler` producers: 4096-logit blocks, four 32-lane warps per block,
one 32-lane merge/tail, stable low-ID tie ordering and fixed 64 list capacity.
The coupled draft uses fixed 64 candidates and the mutable request K at merge;
full and 106299-row CJK domains remain distinct. `sampled_tail_warp:697..802` and
Rust's merge preserve penalties once, then top-k, top-p, min-p, temperature,
ordered F32 sums/quotients/scans, and one Philox draw. Rust's integer mulhi/ten
rounds match `philox_uniform:54..77`, including 64-bit counter carry. Backend
floating-point/transcendental numerical agreement still needs physical tests.

`strata_sampling_request::target_pick` stages a separate per-row consumed-plus-
input-prefix history and uses draw counter start+row, matching
`verify.cpp:2155..2167` and `generate.cpp:11047..11056`. Draft counters are cell+1,
matching the coupled MTP step record (`mtp.cpp:1129..1133`); later draft history
comes from real device proposals. Request parameters remain mutable private GPU
buffers, never kernel/graph keys or persistent content.

`strata_global_controller::accepted/emitted/run` separates accepted input prefix
from delivered predictions. Pending-prefix callback emits verified predictions
before next drafting, while full fault/state/cursor publication remains guarded.
MTP is taught actual next inputs; suffix proposals never override target picks.
This traces the current source flow; it is not an independent numerical oracle
or proof of live target/MTP acceptance and repair.

## Graph/status integration: two real gaps found and corrected

1. Normal `ensure_native_generation_policies` selects grouped checks. The
   `run_global` direct replay guard previously rejected that same selected mode,
   as did GR/GDN/QSA replay adapters. Direct whole-window reuse was unreachable
   in the ordinary grouped path. This does not establish whether the lower
   backend reused some recordings during host-driven re-enqueue.
2. Native GR math writes directly into the sticky model fault, so its separate
   status-fold dispatch count is zero. Model `run_global` already expected
   `[0,36,12]`, but `status_fold_batch::Completed` and diagnostic reporting still
   required legacy `[97,36,12]`. This would reject a healthy native window after
   the GPU work. Older fixture coverage encoded only the legacy expectation.

The coupled fix adds immutable target fold mode to `NativeStrataBranch`, selects
it in the real builder and carries it only from actual joined native replay into
the original Completed receipt. Grouped and individual programs cannot alias.
GDN/QSA adapters still authenticate original prepared owner/sequence/fault/state
receipts, then account their genuinely recorded fold; native GR accounts zero
and validates the matching mode. Whole-window fault reads, exact family deltas,
all-family commit and closing erasure remain mandatory. No naked enqueue or
host counter is numerical/cursor acceptance.

Cold execution and replay now share the exact native `[0,36,12]` completion
inventory. Legacy remains `[97,36,12]`. The normal grouped path is replay-eligible;
independent global experiments, numeric/lifecycle hooks and cancellation injection
remain excluded. Cold count1's ungrouped namespace is separate. A conservative
selected-shape inventory is 126 within the original 128-branch bound, including
all four cold count-one roles. This does not claim an unbounded cache of every
optional diagnostic/full-row/coupled-mode combination in one retained context.
The retained request boundary below closes the cross-request accumulation
question; it does not authorize changing modes during an active request.

CPU evidence is in the linked completion receipt. No new arithmetic kernel, GPU
heap owner, model copy, content persistence, protocol endpoint or Vashti behavior.
The artifact's offline shader branch is permitted; no live model/speed test ran.

## Native program boundaries: ordinary auto-flush connection defect corrected

`WgpuStream::register_pipeline` pushes actual native nodes and increments the
same task count as ordinary WGPU. Its `flush_if_needed` previously flushed at
the ordinary256-task threshold even inside a native request/typed branch.
This is a concrete whole-program defect: the target suffix alone enqueues at
least47 layers x2 GR reads x4 actual GR dispatches =376 math nodes, before any
attention/expert/status work. Thus the ordinary threshold could split a captured
suffix and add host joins/segments. Earlier branch-layout eligibility tests
alone did not establish that the real capture produced the required programs.
This finding comes from the connected source, not a GPU timing observation.

The actual stream now uses the original `TransientIo::reuse_active()` nonce
state and real `Parameters::active` state for automatic-flush policy. An active
native request or branch keeps its explicit boundaries even beyond256 tasks.
This also covers initial uncaptured count1 and prompt work; no fabricated
position-zero typed invocation, new scope/API or cache key is required. Outside
those original states, or outside the native graph, ordinary batching remains.

Actual PLE prefix/input transfer/suffix, target commit/pending export, draft
entry/end, scoped readback, prompt router readback, cleanup and real fence/error
joins remain explicit and independent of the automatic threshold. Holding nodes
does not authorize publication or conceal a failed dispatch. Existing32768-node
and32768-record limits,4096 pages,2MiB templates/4MiB program metadata,64MiB
retained recording metadata, private I/O/admission limits and request erasure
remain unchanged. Actual full-window/all-shape footprint under these quotas
still requires protected execution; no pressure-driven partial replay/fallback
or cap increase is introduced by this fix.

CPU regression uses the actual source policy and typed parameter owners: two
original805-record programs cross three ordinary task batches each, retain
exactly two segment/page inventories and render fresh controls on replay.
Uncaptured request holding, ordinary idle threshold, named commit/draft and
failed-layout clear are covered too. These tests grant no GPU-fence/numerical
acceptance. See the boundary receipt for final source/artifact evidence.

## Retained graph lifetime: cross-request capacity traced

`ResidentAdmission::end_request_io_reuse` consumes the actual inspected erase
receipt before releasing its nonce. The backend's `end_transient_io_reuse`
performs `sanitize_transient_io(true)` and only then `finish_reuse`.
`sanitize_transient_io` refuses outstanding leases, drains the original streams,
completes primary access, and calls `clear_completed_recordings` before erasing
request state. That method refuses live flights/uploads, finishes ordinary and
draft handoffs, drops request scopes, and clears recordings/initialized branches.
`TransientIo::erase_checked` also clears binding/INFO caches, branch parameter
templates and their private invocation, and zeroes INFO pages. `finish_reuse`
requires no buffers, charged private bytes, pending INFO or caches before nonce
reuse. `bind_completed_native_request` authenticates that original reset receipt,
restores the actual route map/usage, checks faults, and rearms a new scope. Its
paired MTP reset joins the original native request before accepting original
private owner erasure; failure retires the context instead of publishing cursor0.

Thus the selected 126/128 inventory is **per request**. Sequential greedy and
coupled requests do not accumulate branch shapes. Original model weights and
kernel programs remain retained, but command recordings and mutable parameter
templates are recorded again for each request. This is an explicit privacy and
request-ownership backend substitution, with an unmeasured recording cost; it is
not a claim of hot command-graph reuse across requests or measured recovery.
Current source: `executor.rs::end_request_io_reuse`, vendor `stream.rs::{end_transient_io_reuse,sanitize_transient_io}`,
`strata_graph.rs::clear_completed_recordings`, `transient_io.rs::{erase_checked,finish_reuse}`,
`strata_parameters.rs::clear`, and `resident_model_public_reset.rs`/`resident_mtp_device.rs`.

## Default dense wide paths: missing coupled producers now connected

Pinned Intel `native_mmvq.dp.cpp::wide_k/try_wide` defaults Q4_K/Q5_K/IQ4_XS to
its wide kernel; `wide_32/try_wide32` defaults Q8_0/IQ4_NL to wide32. Rust already
selected its native Q6_K wide path, but these other formats still used a generic
32-value/lane kernel with FMA and unspecified `subgroupAdd` order. This was a real
selected-path difference, not an optional upstream experiment. Original header
inspection confirms actual target attention/shared matrices in all these formats:
47 shared DOWN IQ4_NL plus layer47 Q8_0, and Q4_K/Q5_K/IQ4_XS attention/shared
roles. Original MTP's nine converted projection matrices are Q8_0. Embedding
formats in that header inventory do not imply GEMV consumers; their distinct
original gather paths remain.

`strata_dense::project` now selects exact real RHS counts1..8 for all six formats;
`project_indexed` uses the same native Q8_0 wide path with its original ID/page
bounds. `strata_native_dense.wgsl` connects original packed weights, existing
single Q8_1 planes, one row/32-lane subgroup and four rows/128-thread block:
Q8_0 has two lanes/block, IQ4_NL one lane/block, and K formats eight lanes/block.
Each lane loads its original weight fields once and applies them across all real
RHS; Q4/Q5 retain separate minima, IQ4 retains its register codebook and signed
scales. Original native contribution grouping and XOR16/8/4/2/1 reductions replace
the generic FMA/reduction path. The superseded generic compute body is removed.
Output/fault owners, graph dependencies and Q8 preparation remain original.

All native dense descriptors use the original padded aligned owner. Q4/Q5's
aligned16-byte loads remain aligned; IQ4_XS's two8-byte reads use the same checked
A2 aligned-vector reconstruction as the already ported Q6_K and wide32 kernels.
The host checks owner size/base alignment; no repack, model duplicate or private
scratch is introduced. `STRATA_ROUND_PRODUCTS` sends these native expressions,
including Q6_K/head variants, through the existing actual registry's FMul
NoContraction postpass. This explicitly fixes backend product-rounding rather
than assuming WGSL/Vulkan and Intel precise-mode compilation choose identical
contraction. Physical numerical agreement remains unproved.

CPU tests independently decode original packed GGUF values for five new formats,
all eight RHS, signed/high-scale/minimum fields and partial block waves; check
actual A2 reconstruction against original bytes within padded owners; and parse,
validate and emit all48 dense plus one indexed specialization, with actual
owner READ/WRITE use and real rounding postpass. See the completion receipt for
the complete selected suite/offline artifact (600 CPU cases/85 suites and
397 offline variants/139 physical checks/72 rounded modules/32 matrix configs).
The prompt dequantizers are real shared-helper consumers: the first offline
artifact caught the removed delimiter dependency; final source preserves all
original factors/codes helpers and directly includes them in prompt Dequant.
This does not establish physical
memory traffic, numerical output, whole-model acceptance or performance.

## Routed expert and prompt matrix consumers: selected source comparison

Pinned `sycl/src/kernels/cuda/iq_kernels.dp.cpp::{Multi,row_dot_multi,row_entries}`
prepares packed weights once for each 4/2/1 group of actual routed entries.
`strata_global_expert_kernels::{role_source,role}` selects the same grouped
consumers over the original VRAM or registered HOST address tables. The generated
single role dispatch contains all three cardinalities; it does not select just
one optional chunk kernel. IQ4_XS has no native `Multi` specialization and keeps
one entry at a time. IQ4_NL DOWN retains two16-value calls per32-value block.
`strata_expert_roles::source` preserves the signed integer IQ3_XXS/IQ2_S
scale/truncation expressions before F32 conversion, IQ3_S odd scales, IQ4_XS
scale-minus32, original Q8_1 activation scales and lane XOR reductions. GU and
DOWN use the same real original route/member map, private planes and PART roots.

The native explicit `contract(off)` statements in this file are attached to
the fused SwiGLU/Q8 producers, at the entries and S26 kernels. Rust's actual
`native_q8` producers have the corresponding rounded-product postpass. This
does **not** prove every role dot product is decorated NoContraction; the role
sources have no such marker. Float backend contraction/transcendentals and
numeric equivalence still require physical acceptance. Do not broaden the
postpass based on the historical completion table's imprecise wording.

The selected prompt path is a separate, coupled consumer chain:

1. Pinned `prefill.cpp` groups real `(token,rank)` routes by logical expert ID,
   preserving token/rank order within each expert. Rust
   `strata_prompt_groups::group` does the same before translating each group to
   its original physical cell. The map retains each original route, so DOWN
   results and final top10 reduction stay in original token/rank order.
2. Native routed-only prefill uses `STAGE=8`, lookahead7 and `DQ=2` dequantized
   images. Rust `strata_prompt_experts::enqueue` connects eight original raw
   slots, two FP16 weight images, both GU/DOWN dequantizers, GU matrix multiply,
   interleaved SwiGLU/FP16 activation, DOWN matrix multiply and reduction. Native
   source releases a raw slot after dequantization; the Rust DOWN dequantizer's
   actual release marker and read/copy intervals allow the same dependency.
3. Native `Gemm::{bf16,f16}` uses oneMKL with BF16/FP16 inputs and F32 outputs.
   Rust `strata_prompt::project` retains original BF16 weights or dequantizes
   the selected quantized dense matrix to FP16, packs the actual matching input
   image, and calls `strata_matrix::dense`. Routed FP16 images call
   `strata_matrix::expert` directly. The old scalar WGSL GEMM is an offline
   diagnostic, not the actual core consumer.
4. Native default `bf16x2_mode=0` does not add a remainder GEMM. Rust's selected
   one rounded BF16 image follows that default. Packed GR BF16 projection has
   its own real consumer; the generic prepacked BF16 rejection does not remove
   the GR path.

Explicit backend/layout substitutions: the Rust FP16/BF16 cooperative matrix
tile is8x16x16 on a16-lane subgroup with F32 accumulation instead of oneMKL's
private tiling/reduction. A masked partial tile still writes only actual rows.
The raw ring advances by logical group, including resident groups, whereas
native staging advances only for nonresident groups. Resident weights stay in
VRAM in both; unused slots may shorten copy lookahead without changing the
eight-slot bound or expert math. Route-indexed Rust DOWN output replaces native
group-contiguous output plus a slot indirection in the rank-ordered reducer.
These are connected substitutions, not evidence of equal physical overlap or
bitwise matrix results.

The selected routed-only ring is not a claim to implement the optional larger
stream-all ring: pinned `ring_slots` enables its default only below10% nonresident
pairs. Even the smallest supported pair is1,510,400B;90% of48x512 such pairs
plus the actual3,644,840,960B dense matrices exceeds a32GiB B70 before remaining
owners. The selected artifact's larger/mixed pairs cannot cross that threshold
on this card. The pinned SYCL CMake target builds `moe_fused_stub.cpp` and does not enable
`STRATA_PREFILL_MMQ`; the `mmq::built()`/fused fallbacks in `prefill.cpp` are
false. Its selected expert products are the FP16 path even for long prompts.
Explicit stream-all/peer/AMD alternatives are not the selected configuration
and are not silently substituted for those products.

Existing selected CPU and offline shader receipts cover original-format role
generation, route/group validation, stage bounds, pointer/read access and matrix
compilation. They do not validate physical products or the whole prompt. The
current audit found and corrected a real additional connected bug: cooperative
expert DOWN stored completion at the old20480-word status-plane offset while
the current4096-row reducer reads40960. It now compiles the shared
`strata_prompt::groups::ROUTES` constant. The actual SPIR-V regression failed
before the fix (20480 versus40960) and passes after it for all eight compiler
layout combinations, tracing the original activation-status read through the
integer addition to the real completion-status store. The rank-ordered reducer
keeps its original shared plane definition. This corrects source/compiled
connection; it does not prove physical prompt products or whole-model execution.

## GR/GDN selected defaults: connected decode and prompt audit

The actual `resident_model_batch::run_global` and `resident_mtp` consumers call
`strata_gr` over the original pending residual/block/injection owners. Pinned
`sycl/src/kernels/cuda/fused_gr.dp.cpp` selects `gr_norm_split()` and
`gr_down_sliced()` by default (explicit zero opts out). The hc check/variant
family is not selected absent `STRATA_HC_CHECK`; CUDA/HIP alternatives are not
additional dependencies of this Intel default.

The connected Rust `strata_gr.wgsl` Norm follows `gr_norm_split_port_kernel`:
256 threads per token/HC stream, identical4t+1024k visits, at most three retained
float4s, original pending residual FMA, weighted values before scale, XOR16..1
and ordered eight-warp sum. Slices follows `gr_down_sliced_kernel`:80 slices of
128 columns, four lanes/row, four aligned eight-BF16 packs shared across actual
rows1..8, XOR1/2 and original324 down/injection rows. Reduce adds all80 slices in
order. Up stages320 low activations/token, retains the same warp/column mapping,
original BF16 dots and XOR, sigmoid mix across four streams and pending residual
FMA. Write materializes the same pending write at PLE/final boundaries. A single
LOW global writer replaces upstream's repeated identical stores; every Up block
still stages the same values locally before its own dots. This avoids racing
stores without moving a consumer or adding a new owner.

A real arithmetic difference found here is corrected: native Up's low activation
is `x / (1 + exp(-x))`, while Rust had `x * sigmoid(x)`. Prompt already divided
for low activations; its injection now spells native `2 * sigmoid(x)` as well.
Division and a rounded reciprocal product can differ; source equivalence cannot
be inferred from their algebraic equality. This review does not claim native
and Vulkan exp/rsqrt or contraction yield physical bit parity.

The actual `resident_gdn_batch::enqueue_native_gdn_operation` uses prepared raw
QKV/Z and the original BF16 alpha/beta producer, convolution/history pair,
recurrent state pair and finite fault/status owners. AB follows Intel
`verify_kernels.dp.cpp::gdn_ab_multi_kernel`:64 threads/two heads, eight BF16
weights/lane, explicit ordered FMA across all real rows, XOR16..1, beta sigmoid
and alpha log1p/softplus with threshold20 and original dt/ssm_a. Rust log1p is an
explicit WGSL backend approximation, not a missing host projection; physical
transcendental accuracy remains unverified.

Conv follows `gdn_conv_l2_multi_kernel`:128 channels/block, original four RAW
causal taps, SiLU, Q/K-only four-warp L2, and final three RAW history rows. Step
follows the selected default `gdn_step_norm_multi_kernel`:512 threads/head,
four row groups each carrying32 state values/column, double-buffered Q/K, ordered
four-group prediction, beta delta, explicit state/readout FMA, four-warp output
norm and gamma/sigmoid-Z. `STRATA_GDN_SPLIT` is opt-in; default accepted-prefix
commit is a separate128-thread/four32-column-tile state replay, already selected
by `strata_gdn::replay_grid`. Verify preserves original prewindow state/history;
original bound accepted-prefix commit repairs/publishes only n. Candidate writes
or a staged count never grant state/fault completion authority.

Prompt `resident_gdn::prompt::enqueue_prompt_full` uses actual full QKV/Z GEMMs
and original AB matrix consumer, then `strata_gdn::prompt::{convolution_full,
recurrence_full}`. It selects128-channel/64-token causal conv tiles, separate
Q/K L2, original128-thread/32-column software-pipelined recurrence and final F16
norm output. Intel `prefill/kernels.dp.cpp` defaults to tiled conv (serial and
CONVL2 are opt-in), `gdn_rec_cols_pipe_kernel` (KEYHEAD opt-in, AMD quad outside
Intel path), and `gdn_out_norm_kernel`. Rust's contiguous token-major H, native
row/value-head state, preloaded next Q/K/V/control tuple and ordered recurrence
match these selected producers/consumers. Output is packed into the original
activation stage consumed by attn_output GEMM. Original paired state/history
roots and all-layer prompt ticket preflight precede publication; no CPU float
state mirror or decode-sized substitute handles a large prompt.

## QSA selected attention/indexer and gate audit

`resident_qsa_batch::enqueue_native_qsa_operation` connects original Qfull split,
Q/K/index-query projection,256-thread norm/RoPE, packed candidate, native
indexer, score/selection,64-cell attention and merge, gate and output projection.
GDN/QSA dispatch geometry remains selected from actual bound layer families.
`resident_mtp` consumes the same selected attention/gate family with its own
original dense INT8 cache and full context; it does not inherit target's sparse
2051-cell cap.

Target candidate/committed ABI is264 words:1024 INT8 codes plus16 F16 scales.
The Rust128-thread pack combines two independent native64-value quantization
groups, preserving separate maxima/stored-half scales, nearest-even codes and
actual full-owner dependencies. Packed32-bit word ownership is a Vulkan storage
substitution, not a change to the64-value quantization scale. Original per-query
causal selection restricts every candidate/cache read. Norm/RoPE retains256-thread
XOR/eight-warp grouping, gamma, the first64 rotated dimensions and original
positions; transcendental/backend rounding remains a physical acceptance gate.

`Kernel::Indexer` follows selected Intel `native_qsa_indexer_append_steps` over
real contiguous positions: round incoming keys through F16, three-row tail,
ordered four-key mean and materialized products, original gamma/norm/RoPE and
retained completed pooled blocks. Snapshot/window owners retain only bounded
checkpoint input for accepted-prefix repair, not a full F32 key history.
`Kernel::DirectScore` follows `qsa_select.dp.cpp::block_scores_multi_kernel`:
256 blocks, eight warps/block, actual <=8 query rows, four positive head scores
with original float4/XOR grouping, incomplete-block1e9 boost and per-query
completed/dead-block choice. The existing Rust ABI omits an empty exact-boundary
block and stores its dead block at the fixed capacity slot; this is a reviewed
layout substitution, not permission to score a future completed block.

Attention/merge follow `qsa_decode_attn.dp.cpp`'s default256-thread chunk kernel:
two KV heads, twelve query heads/KV,64-cell chunks, original eight-value partial
dot/XOR, max/sum, ordered value FMA and stable chunk max/rescale/FMA merge.
Target's fixed33-chunk scratch is a bounded layout substitute for dynamic native
strides. `MtpAttention/MtpMerge` specialize the same body to the full32768 context
and actual packed original K/V page layout, with no sparse target truncation.
Prompt uses32-query scratch tiles with full-query indices for causal output and
selection; it reads the authoritative appended cache only after actual append.

Native decode/MTP `native_qsa::gate` multiplies attention by an already rounded
sigmoid. Native prompt `gate_attn_kernel` does that before one F16 rounding.
Rust had replaced these with direct division; the connected consumers are now
corrected together. Decode/MTP use256-thread blocks/24 blocks per6144-value row,
matching native. Prompt uses128 packed-pair writers/block (256 output values),
with both regular and sticky-fault consumers and both actual launch grids
updated. Pair ownership avoids two threads racing the same32-bit storage word;
it preserves the original contiguous256-output grouping. No output owner,
activation precision or fault/acceptance guard changed.

Source arithmetic, storage and scheduling mappings above are reviewed for these
operator families. Actual physical numerical/state, prompt/count1/multirow,
retained and all-resource recovery acceptance remain required; CPU/compiler
coverage does not substitute for them.

## PLE operators: actual decode/prompt consumers and materialized arithmetic

`resident_ple_batch::enqueue_prepared_checked` authenticates the original prepared
transaction, enqueues original key/value projections, selected native tail, seven
status folds, history status and final finite check. `tail` chooses `native_tail`
unconditionally in the Intel core. That tail uses original BF16 key/value,
1024-thread per-stream key/query normalization,512-thread gate, gated broadcast,
1024-thread norm,256-thread four-tap dilation3 convolution and history. The native
BF16 projection uses ordered weight pairs/two explicit FMA and32-lane XOR/block
reduction. For2560 inputs upstream `mmvf_block_size` selects256; its rows/aux path
requires opt-in `STRATA_MMVF_ROWS`, not an additional default prerequisite.
Prompt uses the same original format projections through its existing rounded
GEMM consumer, then actual PromptNormInPlace/PromptNorm/PromptGate/Broadcast/Norm/
Conv[InPlace]/History. Single writable descriptors retain legal key/result aliases.

Pinned `native_ple_postops.dp.cpp::gate_kernel` explicitly requires materialized
MUL products before ordered additions, with a separate explicit SCALE FMA.
`conv_residual_kernel`/batch use materialized F32 terms from original F16 weights,
assign the first tap, then add taps in order before SiLU and ordered residual sum.
The Rust source already had temporaries/order, but actual raw SPIR-V had FMul
with no NoContraction decoration. A source temporary did not enforce the native
requirement. Source `9a3567e6` marks Gate, PromptGate, Conv, PromptConv and
PromptConvInPlace for the existing actual native registry/offline product pass.
All FMul products are now decorated; complete nonannotation instructions, IDs,
header, original owners and explicit FMA remain unchanged. The unrelated norm,
projection, broadcast and history families retain their own arithmetic contract.

A directed public F32 cancellation witness demonstrates why this matters: two
rounded gate products sum to0 and gate0.5; contracting the second product leaves
a tiny negative residual whose sign changes the native clamped gate by>0.0002.
This is operator arithmetic/compiler evidence, not a claim about an observed GPU
failure or whole-model output. Actual kernel SPIR-V regression covers all five
consumers, raw missing decorations, final exact product IDs, FMA preservation,
idempotence and unchanged complete nonannotation instructions.602 selected CPU
cases/85 suites and397/139/77/32 offline shader/physical/rounded/matrix checks pass.

The original GPU OLD/NORMALIZED/NEXT history remains authoritative. Full accepted
prefixes swap original histories only after joint family fence/preflight; partial
repair reconstructs original nine rows from the original prewindow history and
accepted normalized prefix. The CPU advances two token IDs, not a float mirror.
Original nonce/generation/sequence/state/reset guards remain. Physical numerical,
retained execution and full recovery acceptance still remain unverified.

## Default PLE reader backend: original cache and Linux pool connected

Source `83d690e1` connects `strata_ple_reader` to actual `Session::run` around the
controller. A single request owner serves decode, captured prompt prefetch leases
and synchronous prompt fallback. All use original packed rows from the same
validated table/model binding. Normal engine execution cannot silently fall back
to the old four-worker path; that path remains only for standalone operator probes
without an engine request owner. Full request success requires actual reader/
worker completion and owning cache erasure before retained target/draft reset.

The selected Linux source is `sycl/src/platform/direct_file.cpp`, not io_uring:
its actual `DirectFile::open` calls `io_threads(64)`, despite the earlier generic
comment saying16. The pinned CLI defaults1M cache rows and256 pending slots;
this matches saved-reference allowlisted `--ple-io direct` with no cache/inflight/
sync-submit override. Process environment overrides are not independently
attested. The Rust backend now has one coordinator,64 persistent blocking read
workers,256 aligned two-page slots, original sorted page grouping (including
straddling90-byte rows and short final pages), completion-based scatter and
original eight-way/round-robin64-bit hash cache. It caches packed original bytes,
not a full decoded table. The chosen direct format is actual IQ4_NL
`[160,320001536]`/90B. Default data/keys/replacement owners total98,697,216B.

`kuno_text::gguf::DirectSource` reopens only `/proc/self/fd/<original fd>` read-only
with independent Linux x86_64 O_DIRECT flags; dup/F_SETFL would wrongly affect the
original descriptor. Dev/inode comparison, original model borrow and stat guards
retain the original identity even after pathname replacement. Range/alignment
checks permit only table-intersecting pages; caller verifies each row is within
the actual returned prefix. Complete batch stats are checked before/after reads,
including cache-only requests. Linux x86_64 is the supported direct backend;
unsupported direct I/O fails explicitly instead of silently changing mode.
Original immutable-file/mapped-source obligations remain necessary.

Privacy requires a request lifetime instead of Strata's retained model-lifetime
row cache. The scope shares private endpoint leases only with the actual prompt
producer, rejects wrong model/row geometry/stale leases, and clears main TLS on
success/error/unwind. Stop/disconnect ends keepalive; queued reads and workers
join before descriptor or buffer release. Already entered blocking pread must
return; there is no false cancellation/freed-memory credit. Failure/unwind also
joins the pool, then fails the request. Page slots are wiped after completion and
again after joins. Cache keys/packed bytes/replacement state are volatile-wiped
and inspected while the full original capacities/addresses remain owned. Wiping
plans keep initialized tails clear; popped read-job offsets are volatile-wiped
in the queue before moving them. There is no row/content logging or disk cache.
The default100ms/60s keepalive uses original bounded random public table pages;
read activity rather than cache lookup rearms idle time. All work ends with the
request, so no cross-request row-cache warmth is claimed. GPU model weights stay.

The complete maximum reader HOST reservation is151,453,696B (144.4375MiB), now
included in the actual native `CORE_CACHE_HOST` constructor/admission inventory.
It derives from the largest supported original110-byte row cache (90B for the
selected model),64 finite reader stack allowances plus coordinator,256 aligned
slots/control, and coexistence of two bounded caller plans/raw results. Cache/caller raw/plan Vec capacities are checked and queue occupancy is bounded; no uninitialized tails or uncharged GPU MAIN are
borrowed. This maximum overreserves the90-byte profile explicitly. Existing
prompt/current-next source and model cleanup quotas remain charged too. This
is a source/declared-capacity inventory, not observed whole-model RSS pricing.

CPU evidence:611 distinct selected cases/86 suites, strict core/off binary Clippy;
all eight existing core positional cases also pass. New reader cases cover
original hash/cache replacement, full default owner capacity and actual erasure,
real direct pages/crossings/duplicates/EOF/warm-hit plans, independent flags and
original inode after replacement, queued failure/cancellation/drain, shared
prompt/decode leases, stat change on warm requests, body/coordinator unwind and real worker joins,
and maximum65536-row plans. An external CPU-only audit includes this exact Rust
module and actual original two-shard graph, comparing eight fixed public rows
(including the final table row) cold and warm with the independent GGUF reader;
original metadata remains unchanged and scope joins/erases. It performs no model
inference or GPU/device work. Exact artifact receipt follows in the completion
log. These checks do not establish full physical engine execution.

## Connected PLE graph handoff: launch before the layer1 input join

Pinned `sycl/src/core/verify.cpp::run` launches its graph before `gather_batch`
at1947 (actual `ar_on` arm), then publishes the layer1 ready flag. The doorbell
alternative at2035..2044 gathers after releasing layer0; `pl_launch/service`
similarly separates launch and gather. The prior eager Rust EMB preparation was
a real scheduling gap, not closed by prompt prefetch or MTP overlap.

The current connected Rust path issues request-owned direct reads during
`prepare_deferred`, then submits actual layer0 plus the pending GR FFN write
before `finish_deferred` collects/scatters/uploads EMB. Layer1 PLE and layers1..47/
head follow that upload. Issuance can start earlier than upstream's post-launch
gather; the real read operation still belongs to the same original request,
source descriptor/stat guard and pending PLE transaction. No PLE consumer or
state publication runs before actual rows are ready.

**Necessary Vulkan substitution:** two retained math programs replace the
single SYCL graph's host-ready spin/doorbell. Prefix submits without a CPU join;
only after source collection can the real prefix fence/input upload be joined.
The EMB upload is an unnamed Copy/Fill-only submission, not part of either
retained math program, so a later replay cannot read a stale/wiped upload source.
The suffix consumes original EMB/history/residual/fault owners. No simulated
readiness, uncharged GPU flag or GPU spin was introduced. Cold unrecorded count1,
first capture and retained replay use the same read/submission/layer1 ordering.
Diagnostic paths with additional observations keep their own boundaries and
remain ineligible for direct replay.

A target-only immutable handoff bit separates old one-program diagnoses from
selected two-program keys. The ordinary replay API still refuses multi-segment
branches; the specialized target API preflights exactly two sealed typed original
programs and both actual graph recordings. Continuation requires the same
request nonce, original arena/client, branch, segment1 and every original private
control. A dropped submitted-prefix receipt retires the context; it cannot grant
numerical/fault/state/erasure credit. Suffix joins before creating the existing
Completed receipt, and original all-family observations/commit/cleanup still
follow. Transfer-only detection is restricted to that exact empty segment1
boundary and does not call the mutating parameter-recording accessor.

Actual read tickets/results share a maximum-two-operation quota. The quota
remains held by queued, reading, ready and returned row owners, including after
a caller drops a ticket; only destruction of the final physical owner releases
it. A capacity-one reply permits a ready result to wait for layer1 without
stalling coordinator delivery. Request cleanup joins all real workers and
refuses successful scope return while tickets/results escape. Existing full
reader HOST reservation, source/caller bounds, GPU owners and private accounting
remain in force. Extra segment/page rounding and actual full-model pricing
still require physical admission acceptance under those unchanged quotas. EMB
staging moves its wiping allocation into the original
private upload path, rather than cloning it.

Source substitution is connected; physical overlap, numerical equivalence,
retained execution and whole-resource recovery are not yet accepted. Exact CPU
and frozen-artifact evidence is recorded in the completion receipt.

## Shared/MTP queue and INFO ownership: additional actual call-chain trace

Pinned `verify.cpp::record_window` forks the shared expert to `sh_cs_` and waits
for its join before combining routed and shared results. `mtp.cpp` has its own
default shared branch, likewise joined before combining. Actual Rust target
`resident_model_batch::run_global` wraps `prepare_native` in
`strata_shared_native::QueueScope`, ends that section before routed role work,
marks its original final MAIN node with `release_main`, then folds the separate
shared fault and combines. Actual `ResidentMtp::native_moe` connects its own
separate Q8 planes/fault and the same scope/end/release/fold/combined-output
sequence. Merely compiling shared math is not the connection being relied on.

`WgpuStream::native_strata_shared_boundary` authenticates the real request nonce
and any original target/MTP recording. `Graph::shared_boundary` independently
authenticates MAIN/no-draft versus active original draft scope before tagging
the actually enqueued nodes. `strata_schedule::{Planner,Commands::submit}` uses
the nodes' original byte ranges for RAW/WAR/WAW dependencies and ring release
edges. It submits real compute/copy/shared command buffers with fresh timeline
epochs and a final owning fence that joins both peer tails, including tails
with no later compute consumer.

The two adopted compute queues are reused in opposite roles for MTP's shared
branch. Its shared expert therefore queues behind any already-submitted MAIN
commit; this is an explicit Vulkan substitution for upstream's separate side
stream and may limit overlap. It is not permission to drop that commit, add a
third guessed hardware queue or report physical independence from source tests.

Current INFO copies stay on their original compute lane. Actual
`Graph::upload_parameters` preflights the copy's original read/write intervals,
joins conflicting flights, completes the copy-pool owner before reuse and
submits the current upload. Its actual epoch is attached to the submitted math
header; final drains retain separate INFO-only visibility debt and submit a
real visibility dependency, rather than treating a host copy fence as shader
visibility. Retained recordings own their original descriptor pools/resources,
family, ranges and stable recording ID. Flight completion re-finds that exact
ID and waits its original lane fence before credit/reuse. Exported commit
handles additionally bind original graph, submission and request identity.

`prepare_draft_scope` pins actual verified R and submits the MAIN-to-draft
readiness dependency **before** accepted-prefix commit. Active draft access
checks exclude writers to that R. Scoped draft completion joins its own lane
and actual conflicting peers; ordinary/final closure still drains both lanes.
This trace does not accept every controller/repair transition, full retained
recording footprint or physical Vulkan ordering. Those gates remain open.

## Controller, accepted-prefix state and closure: final source trace

Pinned `sycl/src/program/generate.cpp`'s selected serial serving loop verifies
`[confirmed input, drafts...]`, counts matching draft/target pairs, commits
`a+1` inputs, emits at most `a+1` target predictions, then drafts from the verified
rows. The source explicitly separates consumed input history from emitted
predictions and does not shorten the verification window at the output cap.
`strata_global_controller::{PromptPlan,accepted,emitted,prepare_next_draft,run}`
connects those same transitions to actual target/MTP owners:

- Prompt prefill consumes exactly `n_prompt-1` inputs, in bounded native chunks.
  Its MTP teacher tokens are the real one-position-shifted prompt slice,
  including the final prompt token; target residuals are actual device rows.
  `resident_model_prompt::prefill_prompt_chunk` invokes
  `ResidentMtp::append_prompt_device` inside the same prompt owner. The final
  prompt input uses the ordinary count-one target path, including the same
  Verified/Pending observer and actual draft-readiness preparation as later
  windows. One-token prompts do not manufacture a teacher row.
- The initial prediction is emitted once. Subsequent confirmed inputs and MTP
  proposals retain their original position/binding/token identity. Confidence
  truncates only the usable proposal prefix; suffix/optional lookup chains remain
  target-verified. Counter and consumed-history ownership follow the sampling
  audit above. Bounded MAX8/request headroom guards run before inference.
- `resident_model_batch::run_global` observes actual target prediction/fault
  packets while all original state transactions remain open. It selects the
  accepted input prefix, releases the real output lease, begins the original
  adaptation flight, pins verified R and submits its precommit readiness
  dependency, then submits the retained accepted-prefix commit. Pending output
  delivery and MTP run inside the authenticated original draft scope. Its finish
  and exact exported commit join precede final family/cursor acceptance.
- `resident_native_commit::submit_native_prefix_commit` authenticates all 36
  original GDN, 12 QSA and PLE transactions against proposed count/start and
  accepted count. Native Intel `one_token_self_commit` defaults on: count one
  keeps its already-computed candidate; multirow commits use real runtime
  accepted-count/position controls and the original state/history owners.
  `strata_gdn_replay::enqueue_pending_prefix` restores original convolution/
  recurrence state and replays only the accepted prefix without recomputing
  output logits. `strata_ple_history::enqueue_pending_prefix` selects the real
  accepted history snapshot. QSA restores the accepted index tail and clears
  rejected suffix cells/pools through `strata_prefix_qsa`; it does not publish
  the whole provisional window as accepted.
- Pinned `MtpDrafter::draft` defaults `mtp_catchup_all` off: only the accepted
  rows teach the next draft. Rust `prepare_next_draft` supplies exactly those
  predictions/R rows to `draft_accepted_device_window`, rewinds speculative
  cache coordinates, batches the accepted front, and retains only completed
  proposals. Count-one proposal policy still appends the real front; EOS/cap
  exits invalidate forward/front authority without an unnecessary catch-up.
- Actual family fault/I/O proofs, original-source checks and every family permit
  precede logical target cursor publication. Dropped/failed draft or commit
  guards retire the shared context rather than authorizing resumed inference.
  Output already delivered before a later failure is not a success/state receipt.
- `Session::run` compares actual callback delivery with controller completion,
  erases completed MTP/target private owners, and retains their original reset
  receipts before allowing the next run. New geometry consumes those receipts;
  no constructor-zero substitute is accepted. Failures attempt checked close.
  `Session::close_inner` attempts draft, target, executor, table and root cleanup
  even when another peer fails, and independently audits original source/file
  identity, request I/O and physical-root recovery. Failed/unknown recovery
  retires the context. Reader scopes join and wipe actual request caches before
  returning; fixed aggregate model policy bins contain no request text/tokens.

Explicit state/backend substitutions: Rust keeps bounded INT8 QSA storage with
F16 tail/snapshot owners rather than the native pointer layout; rejected suffix
clearing is additional cleanup, not acceptance of a different prefix. Scoped
Vulkan fences/timeline dependencies replace native stream events. Rust requires
joined original-family/fault/source receipts before logical cursor publication,
even though target predictions may already have been streamed, matching native
output-before-next-draft order. Successful request boundaries erase private
recording/control state while retaining weights and only fixed content-free
policy aggregates. Within-request graph reuse remains connected; cross-request
private recording retention is not claimed. Numerical/storage/driver behavior
of these substitutions is still a physical acceptance requirement.

## Selected engine source-wiring completeness decision (2026-10-17)

**PASS for the selected engine dependency/call-chain review**, runtime source
`fad14f5dba7395a66cc2c06aea649fe1feeb1ca6`, pinned MIT Intel source
`fb58e0dbc8399662c0e47c76578c6e878b14f6cf`. The reviewed configuration is Intel
B70, original DASLab IQ3_S plus original Q2 MTP, `DEVICE_PLAN=1,NO_HOST=1`,
non-profiled serial serving. This decision addresses completion-gate item 3's
source wiring; it does not pass item 4's whole-model correctness/recovery gate
or authorize item 5's speed trials. No installed backend is changed.

| Requirement | Actual Rust owner/consumer and source evidence | Source result |
| --- | --- | --- |
| Original artifacts, packing, vocabulary | GGUF/model binding; original tensor inventory and nine MTP Q8 conversion receipts; bounded `strata_input`/original Rust tokenizer | Connected; original-file receipts retained |
| Loading and CPU/GPU placement | `Session::load`, static preload, ranked `Plan`, original VRAM chunks and mapped HOST complement | Connected; exact owner/loan substitutions above |
| Expert residency/transfers | Original `Sources`, registered tables/read dependencies, prompt staging and joined loan/restore | Connected; selected NO_HOST counters remain zero; GPU-trained promotion is an explicit extension |
| Routing and grouped quantized roles | Actual GPU route/plan, per-role 4/2/1 or scalar native grouping, Q8 planes, PART/reduction | Connected; native format/integer finish audit above |
| Dense projections/output heads | Selected native wide kernels for original formats and whole original target/CJK heads | Connected; compiler/layout substitutions explicit |
| Shared experts | Actual target and MTP separate packed/fault owners, fork/release/fold/combine | Connected; opposite adopted compute peer substitution explicit |
| GR/GDN/QSA/PLE decode and prompt | Selected original family consumers, native reader/cache, full prompt matrix and state owners | Connected; current completion-plane fix and all reviewed substitutions above |
| Prompt and MTP teaching | Full prompt controller, real shifted next tokens and device residuals, ordinary final count one | Connected; bounded shapes/owner overlays explicit |
| Graphs, queues and synchronization | Actual recordings/descriptors/INFO epochs, range hazards, original flights/fences and pending commit identity | Connected; native request boundaries, two-program PLE and Vulkan queue substitutions explicit |
| MTP draft/target verification | Original sidecar/front, compact/full sampler domains, actual proposal packets and target picks | Connected; original positions/history/draw audit retained |
| Accepted-prefix commit and repair | Original full-window transactions, retained commit, accepted family candidates and original source checks | Connected; source trace above, physical state equivalence unproved |
| EOS/cap, output and policy | Separate accepted input count/delivered predictions; real callbacks and bounded suffix/cost policy | Connected; no fake target/route substitution |
| Request privacy/reuse and recovery | Joined wiping reader/request scopes, reset receipt-gated loaded Session and all-peer checked close | Connected; actual GPU/hot/failure recovery still unproved |

The generic Rust request and bounded reader/tokenizer accept owned arbitrary
input internally without logging it. The executable's selected launcher still
uses compiled **public** fixtures; an independently protected private caller is
not accepted by this source decision and remains a separate required gate before
private/production inference. No public fixture output or internal test count
proves that private caller exists. API/UI/service/release work is outside this
engine source decision.

The original source walk accepted `07ed2a2c`; the current revision adds the
public retained-session validation caller below and reviews its shared checked
close wrapper. Selected inference math/placement/scheduling are unchanged. The
current source passes631 selected CPU cases/88 suites, core/off binary Clippy
and normal frozen offline397/139/77/32 checks. These do not prove physical
execution. The pinned upstream remains unchanged. See the latest completion
receipt for exact artifact identity and the remaining gates.

### Actual public retained-session validation caller

Source `fa6e278c` adds `with_loaded_session`, the common load/callback/checked-close
owner now also used by ordinary one-request `execute`. It catches callback
failure/unwind and closes unless that exact Session already attempted close.
The callback receives an exclusive borrow; it returns only content-free
completion. Original process/exclusive-runtime protection remains an unsafe
caller obligation for its whole duration. No GPU/model/session owner is cloned
or reconstructed between requests.

The explicit `two-identical-public` selector runs the same compiled fixture twice
through that original Session. It checks real callback count, exact output-token
equality and EOS/greedy agreement after the first request has completed its real
private reset. The bounded public expected-token copy is wiping, as is the first
output vector before reuse. No comparison tokens enter receipts/logs. One checked
all-peer close still follows; emitted-token equality never grants erasure or
physical recovery authority. The ordinary default remains one request.

`--retained-check` in the protected runner requires `--intel-core`, original Q2
and the existing `--execution-check`; it cannot request private input or unlimited
repeat. The standalone test lifetime is2400 seconds for two requests versus1200
for the ordinary case. It does not change Kuno inference/API deadlines, ownership
quotas or swap/core protection. Normal fixtures request up to512 outputs with
window4/minimum0.5, greedy target sampling and actual core request seed0.
Source `fad14f5d` fixes the runner's previously hard-coded424242 seed metadata
for this core entry; other lab modes retain their separate existing seed. This
reports actual controls without changing inference or retroactively certifying
other historical entries. Physical repeat acceptance is still open.

## Historical remaining acceptance work before the 2026-10-18 device decision

- Hardware/unload/sudo access is resolved. Actual execution corrected shared
  accounting, router capability, implicit scheduler submission, MTP confidence
  owner view, target/language runtime-permit quote and combined replay-role
  interpretation. Current7f4552d2 passes118 selected CPU tests and core/off binary
  lint. Initial target/draft and long generation entry now work; the current
  protected retained-replay retry is running. Wait for actual terminal result,
  then continue remaining acceptance; do not infer success/throughput from source
  fixes. All completed attempts recover physical memory/services/fault counters.
- Run protected whole-model prompt/count-one/multirow completion, then independent
  numerical/target-verification and accepted/rejected-prefix state checks.
  Validate actual cold/capture/replay, two-program PLE dependencies, EOS/cap,
  context boundaries, full retained recording footprint and unchanged quotas.
- Exercise actual loaded-session hot/mixed request reset and failure recovery,
  including original private erasure and physical GPU memory recovery. The
  default public launcher is one request per process. Its new explicit
  `--retained-check` exercises the actual two-request loaded Session and exact
  public repeat equality, but has not run on hardware. Mixed prompts/controls,
  context/failure cases and private-caller acceptance remain additional work.
- Complete the protected private caller gate before private/production claims.
  Preserve all source-selected dependencies while fixing any device failures.
- Only after source wiring **and** whole-model correctness/recovery acceptance
  pass may live whole-engine throughput comparisons begin.

## Selected engine completeness and correctness decision (2026-10-18)

**PASS for the selected, connected Intel engine and its fixed-public throughput
trial.** This extends the 2026-10-17 source-wiring decision with actual protected
device evidence. It supersedes historical pending numerical/completion/repeat
gates above for this configuration, not the exclusions below. Runtime source is
`299579b52f9e4b9d8feaaaedb41eec4c193fcb83`, pinned MIT upstream remains
`fb58e0dbc8399662c0e47c76578c6e878b14f6cf`. The complete chosen inference path
remains selected together; no coupled component was removed for a timing result.
No live speed test preceded this decision.

| Gate | Current evidence and decision |
| --- | --- |
| Complete source/dependency connection | Earlier requirement matrix and consumer audit pass. Comparison against frozen runtime shows no unreviewed removal. Exact loading/residency, original mapped HOST complement, real experts/router, native families, graphs/queues, original Q2 MTP, target verification/commit and closure remain connected. |
| Original-weight numerical checks | `strata-fp16-preserve-coupled-check-20261018/numerics.json`:74 cases pass:48 GR stages,18 packed-dense/Q8 cases, full248,320-row physical head,32 embedding rows spanning all16 original pages,6 BF16/FP16 cooperative products at counts1/8/35. Original CPU decoding/reference and tolerances remain unchanged. |
| Complete actual model | `strata-fp16-preserve-retained-20261018/intel.json`: two actual requests on one loaded Session, identical362-token output/EOS,110 windows/31 repairs in the reported second request; real reset/rebind after the first. Human-readable unsaved public tail is coherent. |
| Other prompt and output stop | `strata-fp16-preserve-prose-20261018/intel.json`:512 outputs without EOS,194 windows/83 repairs, coherent visible prose. Earlier explicit16 cap also passes on the unchanged cap/control implementation; this new512 cap and EOS run use the current numerical artifact. |
| Request and terminal recovery | Both current full runs erase actual target and draft request owners, complete all-peer close and restore Intel50,688,000B/NVIDIA compute0, all HTTP200 and normal admission with no recovery latch. Classified fault deltas all0. Prior strict numerical failure also completed checked close/recovery. Active systemd receipts show swap0/core0 and the original48.2GiB cap. |
| CPU/compiler checks |633 distinct selected cases/88 suites, strict core/off binary Clippy,397 offline variants and32 cooperative matrix configurations pass. Installed `spirv-val` cannot parse BF16 capability5116; the pinned matrix-capable validator passes the original complete set. |

### The two numerical fixes and necessary backend controls

The mapped original embedding uses aligned original `vec4<u32>` source reads
(`b6744323`). Every checked value matches canonical CPU decoding exactly; ordinary
scalar direct-page control also remains exact. Its lower-level physical scalar
read discrepancy is unisolated, so this is a validated selected-path correction,
not a general claim about scalar buffer reads.

`f0707ddb` adds DenormPreserve16 to actual ordinary-FP16 cooperative SPIR-V only;
BF16 types are excluded. The actual B70 FloatControls property is queried before
module creation and unsupported preservation is rejected. `299579b5` couples
the SAME pure transformation into native replay's own VkShaderModule words, not
only the ordinary HAL module and offline validator. The original matrix kernels,
weights, packing, shapes, accumulation and CPU reference remain unchanged.
The strict FP16 error drops from29 failing outputs at count8 to zero at all tested
counts; maximum matrix error is4.2915344e-6. The additional FTZ oracle now differs,
as expected; it never replaces the IEEE reference or grants acceptance. The
original failed receipts remain failed. Supported floating-point control is the
explicit Vulkan substitute needed here for the pinned Intel precision contract;
no arbitrary tolerance change or alternate expert/model supplies the pass.

### Exact timing permission and limits

`apps/kuno/benchmarks/text/strata-core-public-acceptance.json` permits only frozen
SHA256 `852a0dbd7dbdd11a3a1eb8e3447a9006dbe7fae2b56d8d3ecd1fb3ccd7c473e1`,
source299579b5, original requested core features, code fixture/Q2 and two identical
requests on one loaded Session. The runner rejects changed options/artifacts.
Four synthetic observer tests cover incomplete runs, bad ordering, failed repeat,
missing erasure, unhealthy admission/services and residual/faulted GPUs. The
observer drains and joins a child before rejecting invalid timing. Only successful
completion/repeat and root physical recovery can write a separate throughput
receipt. It never changes the frozen execution-check receipt's static flags.

The last-prompt-inclusive interval includes ordinary final prompt input, first
prediction, target/MTP windows and host controller/delivery. It excludes request
erasure and final model close. It is NOT a pure GPU kernel or cache-free token
rate. Prefix/preflight, runtime setup and request-before-erasure intervals are
reported separately. Response text and tokens remain transient; metadata contains
counts and numeric durations, no contents, prompts, seeds from private requests,
routing trace or content hash. No repeated Ollama baseline is required.

This passes completeness and representative correctness for the chosen engine,
not universal numerical equivalence or production readiness. Independently
protected arbitrary/private callers, mixed requests on one retained Session,
physical maximum-context stress, broader stochastic controls and long soak remain
additional acceptance work before production claims. There is no bitwise whole
model comparison with Strata/CPU, executed-code quality test, public API/UI change,
installed native backend or release acceptance. Those limitations do not indicate
a missing selected source dependency and must not be represented as already tested.


### First accepted whole-engine throughput (2026-10-18)

`strata-accepted-retained-throughput-20261018/throughput.json` is emitted only
after the same actual362-token repeat/EOS, target/draft erase, checked final close
and clean root after-state. Runtime remains frozen299579b5/SHA852a0dbd; observer
was committed at9666324b before this first speed run. The pinned-source selected
engine was complete and accepted before timing started.

| Interval | First request | Second request, same loaded Session |
| --- | ---: | ---: |
| Prefix/preflight |5.4624s |5.2670s |
| Final-prompt-inclusive generation |18.3003s |17.5652s |
| Delivered outputs including EOS |362 |362 |
| Generation delivery rate |19.7811 tokens/s |20.6089 tokens/s |
| Entire request before erasure |23.7627s |22.8322s |
| Entire request delivery rate |15.2340 tokens/s |15.8548 tokens/s |

Selected runtime construction from its trusted `target` phase to first request
is101.1493s. This excludes earlier file/tokenizer/device setup and is not a full
process cold-start metric. No request erasure or final model close is hidden
inside the generation interval. These are two identical fixed-public requests,
not a broad prompt-distribution benchmark or an apples-to-apples new Ollama
comparison. The human's accepted10 tokens/s reference is for a different quant;
we do not claim a measured2x comparison.30 tokens/s remains the performance target.

Physical recovery is independently checked: Intel50,688,000B, NVIDIA compute0,
all HTTP200, no active/waiting/paused/recovery admission and unchanged zero
classified faults. Active-unit properties confirm swap0/core0/48.2GiB. No service
installation, reset/reboot, model substitution or private content persistence.

Next performance diagnosis should sample the native graph's ALREADY COLLECTED
aggregate capture/wait/submit/replay counters at the same three static request
phases. That distinguishes CPU recording cost from GPU/fence time without
serializing every kernel with timestamps. Preserve the complete engine. Do not
return to testing and abandoning isolated coupled components.


### Fixed-public native graph counter diagnosis (2026-10-18)

Runtime candidate `b7cb8e85f0f70ec7e913dc071e835bc93aed830c`, frozen artifact
`strata-public-phase-counters-20261018`, SHA256
`aa12c4732d21b269beac5745ba876a6df6ca0034607daa9e6a8f93299e00bb9e`.
The sole native-source difference from accepted299579b5 is20 added lines in the
fixed-public `daslab_language_probe.rs` caller: read existing process aggregates
at preflight/final-prompt/erasure and serialize at most6 static snapshots. It
adds no GPU query, math, owner, scheduling or resource-queue work. Strict core
Clippy and changed-file formatting pass. All original selected components remain.

Protected `strata-public-phase-counters-retained-20261018` exits0 with identical
362-token/EOS outputs on one loaded Session,110 windows/31 repairs in the reported
second request, both request erasures and final close. Independent root after-state
checks confirm the same physical baseline, all HTTP200, normal admission and
unchanged zero faults. Active-unit swap0/core0/48.2GiB protection is recorded.
This did not use `--throughput-check`; the exact accepted-artifact manifest is
unchanged and still refuses this different artifact for throughput claims.

`phase-counter-deltas.json` derives only monotonic aggregate differences from the
six actual successful snapshots (original `intel.json` remains unchanged):

| Phase | Submissions | Replays | Recorded compute dispatches | Capture counter | Native fence-wait counter |
| --- | ---: | ---: | ---: | ---: | ---: |
| First prefix/preflight |1466 |0 |33849 |0.4341s |3.9270s |
| First generation |2076 |1964 |344298 |0.1797s |15.8513s |
| Second prefix/preflight |1468 |0 |33850 |0.4393s |3.9457s |
| Second generation |2077 |1964 |344346 |0.1704s |15.8413s |

Source `strata_graph.rs::credit` counts actual compute nodes/submissions and
classifies graph size. The second generation has111 large (>=2048 nodes)
submissions,103 replays of those large graphs. `CAPTURE_NS` covers the non-replay
recording path; it does not cover all CPU work or signature generation on hits.
`WAIT_NS` covers actual native lane-fence waits; it is not a pure GPU kernel
timer. These are not additive exhaustive categories. Still, the substantial
native wait and high replay fraction make CPU command recording a poor first
optimization target. No new per-request wall rate is calculated from counters
or guessed from log timestamps. The accepted uninstrumented reference remains
19.7811/20.6089 tokens/s on299579b5.

Next useful work: connect one bounded fixed-public operator-timestamp sample to
the ACTUAL `strata_global_controller.rs` verification window, preserving original
commit/draft lifetimes and all components. Existing arming code in the later
legacy `daslab_language_probe.rs` loop is unreachable for this selected engine.
Both `strata_engine.rs` and `strata_global_controller.rs` deliberately reject
`profile_native_target()` today; merely enabling the runner flag does not create
a valid selected-path profile. Scope/restore the flag on errors, retain genuine
recording/query-pool ownership and collect only static operator aggregates after
real fences. Profiled kernels serialize for timestamps and must NOT supply an
uninstrumented throughput claim. Use that result to choose the next substantive
GPU change; do not return to unrelated API/cancellation work or remove coupled
upstream components after an isolated slowdown. No more Ollama baselines needed.

All GPU runs in this checkpoint are terminal and recovered. No source install,
private caller, frontend, public API, GPU reset/reboot or release changes. The goal
remains active;30 tokens/s remains unmet. If protected sudo expires, stop cleanly
and document rather than trying privilege workarounds.


### Actual selected-window operator profile and next GPU change (2026-10-18)

The first profiler source`a545c0ca6875d09ba0e7bf6990ed1683b1f4f109` fails strict
`native branch incomplete parameter layout` during a mid-walk shared boundary.
Output`strata-selected-operator-sample-check-20261018` exits1/RecoveryRequired,
NO success receipt and NO operator/performance acceptance. Checked private close
also reports failure after I/O retirement; do not call it successful erasure. The
outer protected process exits and physically recovers both GPUs/services with
unchanged zero faults. This was a fixed-public fixture, not private inference.
The accepted normal frozen299579b5 is unaffected; no installed service changed.

Coupled source`f38629a9861026b388728aca377d554254a11281` corrects the actual issues:

- Shared begin/end needs admitted branch identity, not a claim that the entire
  sealed INFO inventory is already rebuilt. `Parameters::active_identity` and
  its I/O wrapper carry original nonce/family only. Submission still calls the
  original strict `recording()`/`submitted()` checks. A real vendor test rebuilds
  a sealed branch and rejects submission until all records AND pages are present.
- Scope GPU profiling only until actual `PrefixPhase::Verified`, after the
  original target read has completed. Pending commit export and next drafting
  keep their normal protocol; the existing profiling exclusion there stays.
  An owning guard also restores the selector on return/error/unwind.
- Both normal direct replay and ordinary rebuilt-node replay exclude timestamped
  recordings. A sampled query recording cannot silently remain instrumented in
  later normal windows. The original recording/fence/query-pool owners stay real.

This is diagnostic wiring, not a numerical/model/placement change. Builder freezes
`strata-selected-operator-coupled-20261018`, mode0500, SHA256
`a7f1b417d97b1170a12554336ca4de79784f5a011007a4a9699d048a6c50df3c`.
All28 actual vendor parameter tests pass using the existing error-only test
wrapper and the current cargo-resolved runtime rlib. Core strict Clippy and
changed-file Rust formatting pass. The runner rejects profiling combined with
retained/throughput or alternative fixture/options before artifact/service work.
No whole633-case rerun is claimed for this diagnostic-only patch.

Protected`strata-selected-operator-coupled-check-20261018` exits0 with362 outputs,
EOS,110 windows/31 repairs, actual target/draft request erasure and final close.
It displays the coherent fixed-public response without saving it. Root after-state
is independently accepted: Intel50,688,000B, NVIDIA compute0, all HTTP200, normal
admission/no recovery latch and unchanged zero classified driver faults. Active
properties confirm swap0/core0 and the original48.2GiB cap. All GPU runs are terminal.

One actual target verification suffix is sampled (round8); it contains the47
post-PLE target layers. Static operator aggregates are collected after real
fences; no prompt/token/activation/route trace is stored. Total126.45584ms across
all sampled operators; `operator-summary.json` is derived from the unchanged
`intel.json` counters:

| Operator | Calls | Serialized GPU time | Fraction |
| --- | ---: | ---: | ---: |
| StrataGlobalNativeDown |47 |32.1633ms |25.43% |
| StrataGlobalNativeGate |47 |26.7151ms |21.13% |
| StrataGlobalNativeUp |47 |26.5893ms |21.03% |
| StrataPackedDenseQ8 |294 |10.2411ms |8.10% |
| StrataGrUpMixCommit |95 |4.6247ms |3.66% |
| StrataNativeBf16Multi |96 |2.9961ms |2.37% |

The three expert roles total67.59% of this sampled target suffix. Timestamping
serializes nodes, excludes unsampled prefix/MTP work and is not an uninstrumented
whole-generation decomposition. It nevertheless supports prioritizing packed
expert projection/memory access over CPU graph recording or BF16 GEMM changes.
The accepted throughput artifact/manifest stays299579b5/SHA852a0dbd; no new tokens/s
result is claimed for this profile build.

**Next substantive candidate:** inspect and remove the selected expert roles'
extra activation workgroup staging. `strata_global_expert_kernels.rs::role_source`
builds actual 4/2/1 entry loops through `strata_expert_roles.rs::source`. Current
GU stages all8*720 u32 input-cache cells (23,040B/WG), even for short groups.
DOWN reserves8*180 cells, while a chunk uses at most4. In upstream
`sycl/src/kernels/cuda/iq_kernels.dp.cpp::row_dot_multi/row_entries`, `Multi::acts`
reads the already-quantized activation directly for that entry. This SLM staging
is a recorded Rust/Vulkan layout substitute, not a required original dependency.

A coherent no-SLM candidate must change ALL paired consumers: remove the selected
activation shared declarations/copy loops, make each member's frame the actual
original plane base, read both packed codes AND scales from their original plane
locations, and remove chunk barriers that protected only that now-absent shared
tile. Keep original route IDs, original GPU/HOST weight pointers, 4/2/1 grouping,
integer dot products/finishing, same per-lane F32 order and XOR reductions, status
publication and full runtime/privacy ownership. Preserve shared codebook handling.
Restrict this change to the selected global path; legacy source builders stay
explicit. Prefer a small helper taking `(frame,relative_word)` over logical-index
division/remainder for every input read. Gu packed0..639 maps to original frame
`64+token*1440`; Gu scale640..719 maps to that frame+1280+(word-640). Down packed
0..159 maps to frame`128+route*360`; Down scales160..179 map to that frame+320+
(word-160). All original full-owner/route guards remain. Validate the actual
compiled source and whole selected engine before adopting a new throughput hash.
This may improve occupancy/input traffic; no gain is assumed before measurement.
No more old Ollama baselines, API/cancellation work, foreign engine, easier model
or removal of coupled port components is needed. The30 tokens/s goal stays active.


## Direct expert activation candidate and GPU handback (2026-10-18 local)

Human requested the GPUs back at this checkpoint. No further whole-model or
speed run was started. The goal remains paused. The accepted runtime and its
19.7811/20.6089 tokens/s results remain unchanged.

Candidate source `dcd7326e2eaefcd7b21c69e6f9243967069c7e1b` implements the
paired direct-plane changes described above: selected global GU/DOWN roles omit
the activation SLM declaration/copy/barrier and consume original packed words
AND scales directly through per-member original plane frames. All real expert
addresses, 4/2/1 groups, integer finish/reduction, route/status guards and owners
remain. Legacy non-global builders explicitly keep their previous staging path.
The candidate is not yet accepted for whole-model correctness or performance.

Immutable build `strata-direct-expert-input-20261019` completed with SHA256
`6a73f7bd8e698bb91f06823a34d26377893b06e46764992f4544e0e2f9558f9d`.
Pinned matrix-capable offline validation exits0:397 shader variants and32
cooperative matrix configurations. Log:
`strata-direct-expert-input-shader-validation-20261019.log`.
Both actual global source tests pass (2 tests,0 failures): the ten compiled
expert-role modules validate with no activation workgroup storage/barriers;
all13 role/fetch schemas preserve their original address bindings. Log:
`strata-direct-expert-input-source-tests-20261019.log`.
All CPU build/validation/test processes are terminal at handback.

Read-only root inspection
`strata-direct-expert-input-checkpoint-host-20261019.json` at00:11:21UTC
(2026-10-19) confirms Intel50,688,000B used; NVIDIA compute processes0,
desktop memory520MiB; Kuno/Ollama/Vashti HTTP200; admission active=false,
waiting=0, paused=false, recovery_required=false. All classified Xe faults,
timeouts, wedge/reset failures and NVIDIA Xid counters are0. No GPU reset,
service restart, driver change, installation, or frontend/API change occurred.

Resume only when GPUs are available again: finish strict core Clippy as needed,
then run the frozen candidate through protected retained code execution, exact
repeat, request erasure, close and physical recovery. Only after correctness
and the source audit pass may the exact-artifact throughput allowlist advance;
preserve the accepted299579b5 baseline and measure the same retained workload.
Neither offline validation nor the address equivalence argument proves a speed
gain. Do not time this candidate before that acceptance, and do not roll back
coupled port dependencies merely because an isolated timer regresses.


## Direct expert products: strict rounding diagnosis (2026-10-18 local)

User returned both GPUs and refreshed sudo. Initial preflight
`strata-direct-expert-input-retained-20261019` rejected BEFORE admission/load:
a cached idle Kuno image worker held17,637,675,008B on Intel. Graceful
`systemctl restart kuno.service` evicted that owned worker. The immediate
post-restart inspection raced listener startup and failed; the subsequent
`strata-direct-expert-input-resume-ready-host-20261019.json` confirms allHTTP200,
Intel50,688,000B and NVIDIA compute0. No driver/GPU reset or installation.

`strata-direct-expert-input-retained-clear-20261019` then completed two identical
491-output/EOS requests,151 windows/44 repairs, real erasure/close and clean
physical recovery. It was NOT timed or accepted: historical baseline produced
362 outputs. The original frozen299579b5 control
`strata-direct-expert-input-original-control-20261019` reproduced362/EOS,
110 windows/31 repairs and clean recovery on this same system.

Source55294dda adds a bounded original-weight/synthetic-input comparison to the
existing protected numerical harness. It invokes ACTUAL global physical-address
expert shaders with both original staged and direct input modes, same real
weights, same immutable Q8 planes, complete private owners and real indirect
read dependencies. It tests10 roles/formats at counts1/3/7/8, including real
4/2/1 groups and multiple chunks. Only numeric mismatch metadata persists.
No generated text, activations or route traces are saved. The old staged
variant is a diagnostic control; production dispatch stays direct.

`strata-direct-expert-input-parity-check-20261019` correctly FAILED on one
Q2_0 DOWN result at count3:1 bit mismatch, max absolute difference
1.4901161e-8, with equal status. Preceding cases matched exactly. The failed
receipt remains failed, with checked cleanup and normal physical recovery.
The actual expert sources lacked STRATA_ROUND_PRODUCTS, so the existing
NoContraction postpass was not being invoked for them. The result is consistent with a different contraction choice when activation
storage changed; a changed response alone did
not identify wrong plane addresses.

Source883d584b explicitly selects separately rounded products for BOTH actual
global expert input modes as a Rust/Vulkan precision control. The pinned file
has explicit contract(off) pragmas on the Q8/SwiGLU producers; this change must
not be described as proof of a role-local pragma on every upstream dot product. The shared existing postpass
marks real FMul results NoContraction and leaves explicit FMAs unchanged. No
reference, tolerance, packing, source address, row-entry grouping or verification
rule was weakened. This corrects a previously untested expert precision gap in
the representative numerical audit, rather than removing a coupled dependency.

Frozen `strata-direct-expert-rounded-20261019`, source
`883d584b98fc30922f2272e8eb4e1228e4be8f46`, SHA256
`32bd4bb33333f6e56d79a4d26d13de139144c1a8e839c05f1b6d9388f4fe6c1c`.
`strata-direct-expert-rounded-numerics-20261019/numerics.json` passes114 cases:
40 expert cases have ZERO bit/status differences and max_abs0; the original74
cases also pass unchanged. Complete private cleanup and physical recovery pass.
Pinned offline shader validation passes397 variants/32 matrix configurations.
Full-model retained validation and throughput acceptance are still pending at
this intermediate checkpoint; baseline20.6089 remains the only accepted warm
rate. The goal remains paused; current work proceeds under the direct resume.


## Rounded direct-input candidate acceptance (2026-10-18 local)

PASS for fixed-public retained engine measurement on frozen883d584b/SHA32bd4bb3
identified above. Full selected loading/placement, original GPU and mapped HOST
weights, real experts, native quant/routing/dense/graph families and target/MTP
verification/commit remain connected. The only production changes from the
previous accepted path are paired activation staging/consumer removal and the
original separately rounded expert products. Source55294dda also adds the
bounded fixed-public diagnostic; ordinary requests never execute it.

`strata-direct-expert-rounded-retained-20261019/intel.json` completes two same
loaded-Session requests, identical362 outputs/EOS,108 windows/32 repairs in the
reported second request. Both actual target and draft owners are erased, final
close completes, all services/admission recover, Intel returns50,688,000B and
NVIDIA compute0; all classified driver fault counts remain0. Active protection:
MemorySwapMax0,LimitCORE0,MemoryMax51,745,128,448B,RuntimeMax40min. Visible
unsaved public tail is coherent; no executed-code quality claim or bitwise
cross-artifact token comparison is made from equal output counts. The prior
build's110 windows/31 repairs differs, consistent with the precision correction.

All114 numerical cases, including40 BIT-EXACT actual staged/direct expert
comparisons, pass without relaxing any check. Both targeted actual source tests,
strict selected-core Clippy and397 shader/32 matrix validations pass. The
remaining private/production/max-context/soak exclusions of the prior audit stay.
The timing manifest advances only to this frozen artifact, original code/Q2
fixture and two identical requests; prior299579b5 acceptance is retained as
`previous_acceptance` and its20.6089 warm result remains historical baseline.
Timing has not yet been collected for the new artifact at this decision.


## Rounded direct-input throughput and joint GU follow-up (2026-10-18 local)

`strata-direct-expert-rounded-throughput-20261019/throughput.json` passes on
accepted883d584b/SHA32bd4bb3,2 same-session identical362-output/EOS requests,
108 windows/32 repairs in the reported second request. First generation
17.689718604s=20.463864243 tokens/s; warm generation16.990734607s=21.305729762
tokens/s. Warm prefix/preflight5.265295616s; full request before erasure
22.256030223s=16.265254692 tokens/s. Target-phase setup138.278999025s excludes
earlier process/file/device setup. All timing scopes/exclusions stay unchanged.
The observed warm rate exceeds the prior20.6089 trial modestly; no repeated
statistical confidence or apples-to-apples new Ollama comparison is claimed.
Recovery: Intel50,688,000B, NVIDIA compute0, normal admission, allHTTP200,
all classified driver fault counters0.30 tokens/s is still not met.

Next coupled candidate follows original `native_gu_kernel`/`launch_gu_port` in
`iq_kernels.dp.cpp`: ONE dispatch covers the gate/up row domain (2*n_ff),
with each workgroup executing just one projection. Current Rust previously
issued the same role work in two sequential dispatches. This is an explicit
launch-layout improvement, not removal of a mathematical dependency. The
combined shader shares original guards/descriptors/address/table helpers and
selects gate or up uniformly by row-grid half; each half retains original
4/2/1 chunks, input planes, complete owners, separate PART offsets/stamps and
product rounding. The coordinator issues it once before the original SwiGLU/Q8
consumer, which waits on both results. Actual numeric cases compare the joint
dispatch against BOTH separate direct roles using DIFFERENT authentic original
gate/up weights; no same-weight shortcut can conceal swapped role pointers.
No throughput acceptance is granted for this new candidate before validation.


## Joint GU candidate acceptance (2026-10-18 local)

PASS for the selected fixed-public retained measurement, frozen source
`0b76f8bb47328fc249ed1af5eea7cb119f6ed0ab`, artifact
`strata-joint-gu-entry-20261019`, SHA256
`91a23e785fb43c3a3ae42fa5e1aaea549d0be84b515c8b345f97a16b7176f523`.
Sourceb91bcb33 implements the coupled row-domain/coordinator change and16
additional actual physical joint-versus-separate original-weight cases. Its
first offline validation rejected entry-only builtin attributes on helper
arguments before ANY GPU execution.0b76f8bb passes those builtins as ordinary
arguments to regular helpers; the final entry remains the sole compute entry.
The initial source test also failed as expected; its replacement passes both
targeted tests over17 schemas/14 expert roles. Do not reuse the rejected build.

`strata-joint-gu-entry-numerics-20261019/numerics.json` passes130 cases:74
original numerical cases,40 bit-exact staged/direct controls,16 bit-exact joint
GU controls with distinct original gate and up weights. All bit/status
differences are0 and complete owning tails are compared.401 offline shaders/32
cooperative matrix configurations and strict core Clippy pass.
`strata-joint-gu-entry-retained-20261019/intel.json` passes two same-session
identical362-output/EOS requests,108 windows/32 repairs, target/draft erasure,
final close and root physical recovery: Intel50,688,000B,NVIDIA compute0,
allHTTP200, normal admission, all classified driver faults0. Active swap/core0
and the same48.2GiB cap observed. Selected full loading, actual source residency,
real expert routing/math, all graph/queue/state/target-verification dependencies
remain connected. Original arithmetic is retained; GU simply occupies one
40-row-workgroup dispatch instead of two20-row-workgroup dispatches.

The exact-artifact manifest advances only after these gates. Prior883d584b
acceptance and its21.3057 warm rate remain nested in previous_acceptance and
unchanged historical evidence. No private-caller, production/API, maximum-context
or soak acceptance is added. New timing remains pending at this decision.


## Joint GU measurement and eight-input request accounting (2026-10-18 local)

`strata-joint-gu-entry-throughput-20261019/throughput.json` passes accepted
0b76f8bb/SHA91a23e78:362 outputs each,108 windows/32 repairs. First generation
17.932528906s=20.186779115 tokens/s; warm17.248659086s=20.987138664 tokens/s.
Warm prefix5.227776613s, full before-erasure22.476435699s=16.105756484 tokens/s.
Target-phase setup101.529923409s. Clean after-state and zero driver faults.
This is near the separate rounded GU trial21.3057, not evidence of a speed gain.
Keep the complete coupled one-dispatch schedule rather than discarding it on
this isolated small regression.30 remains unmet.

Source45a8eb6a gives explicit --intel-core --eight builds a genuine eight-input
public request, instead of merely reserving MAX8 buffers while requesting4.
Ordinary core builds still request4. speculation_window is now recorded in
content-free public completion metadata. The engine already accepts the
request Config; this is a CONFIGURATION trial, not a new GPU math claim.
The legacy launcher label supplies four-lookahead backend flags; the actual
native Request config is independently explicit. The unchanged math has130
prior operator cases at counts1/3/7/8.401 shaders/32 matrices and eight-feature
strict Clippy pass on45a8eb6a.

strata-native-eight-request-retained-20261019 FAILED with Q6 verifier head
actual completed counters. No completion receipt or speed. Cleanup reports
already_retired=true: do NOT claim software erasure acceptance. Process exit
and root inspection restore Intel50,688,000B, NVIDIA compute0, normal admission,
allHTTP200 and zero driver faults.

The actual head contract/producer supports8, but completed-member accounting
still capped at4*prepares.90968c5a uses the SAME admitted capacity:3 when the
extended-row context is inactive, otherwise selected verify_capacity (4 or8).
Unselected work, missing/extra tiles, fewer members than prepares, overcapacity
and arithmetic overflow still reject; no counters are overwritten or skipped.
The actual existing unit test covers3/4/8 boundaries and invalid capacities.
Its actual functions/test are selected verbatim from executor.rs into a small
Rust test linked to the same Cargo-resolved kuno_text rlib.1 test passes, log
strata-eight-head-counter-actual-source-test-20261019.log; sourceSHA256
4829cf8a50b39c687a85114f7c1ab88fbf44d98563e1fd22c6423061405517a8.
This is not a whole-binary test-suite rerun.

Frozen strata-native-eight-counter-20261019, source
90968c5aa70956b3bbd66b809a5f6b7f27ecc067, SHA256
7bba592cc2c539ae733e7ec11e81b2e40d499a20023119b4b99638d31806f7d9,
passes strict eight-feature Clippy and401 shaders/32 matrices. Full retained
validation is running; no eight-window throughput is yet accepted.


## Eight-input configuration acceptance (2026-10-18 local)

PASS for fixed-public retained measurement on90968c5a/SHA7bba592c above,
features strata-rust-port,strata-intel-core,strata-eight-window. Same seed0,
code prompt,Q2 MTP,greedy/minimum0.5, but Request Config is EIGHT instead of4.
strata-native-eight-counter-retained-20261019/intel.json passes two identical
362-output/EOS requests on the same loaded Session,81 windows/50 repairs in
the second report, explicit speculation_window8. Target/draft erasure, final
close and physical recovery pass with Intel50,688,000B,NVIDIA compute0,allHTTP200,
normal admission and all driver faults0. Active swap/core0 observed. Unsaved
public tail is coherent. Equal token count is not a cross-artifact token proof.

The kernels and selected core max8 ownership/math do not change with this caller
configuration: the earlier130 original-weight cases cover1/3/7/8; they were NOT
rerun on this artifact. This artifact's401 shader/32 matrix validation, strict
eight-feature Clippy and actual-source admitted-counter unit test pass. No
capacity/tile/status/erasure check was relaxed to accept the earlier failure.
Native8 numerical/state/mixed/private/long-soak coverage is still representative,
not universal. No private caller/API/service installation is accepted.

The exact-artifact timing manifest now explicitly includes speculation_window8.
The outer runner verifies that field after completed recovery before writing
throughput.json and records the chosen window with the numeric rate. Frozen
four-input acceptance remains in previous_acceptance. All other request/timing
scopes stay unchanged. No eight-input rate has yet been measured at this decision.


## Eight-input result and adaptive-cache acceptance (2026-10-18 local)

Eight-input accepted90968c5a/SHA7bba592c throughput:
strata-native-eight-counter-throughput-20261019/throughput.json.362 outputs per
request,81 windows/50 repairs; first20.185683657s=17.933502088 tokens/s,
warm19.092841012s=18.959986090 tokens/s. Warm prefix5.280746620s, full request
24.373587632s=14.852142633 tokens/s; target-phase setup98.012953067s. Clean
physical/services/admission recovery and zero faults. Eight inputs reduce window
count but increase work/repairs enough to be slower. Four stays the ordinary
fixture default; the implemented eight-request mode and accounting fix remain.

Next trial enables the ALREADY implemented GPU-usage/adaptive two-way cache
extension, retaining four-input requests, direct expert input, explicit product
rounding and joint GU. This is beyond pinned NO_HOST's zero counters: real GPU
route counts now feed the same bounded every4-window rank/copy/publication path.
Original expert bytes swap both ways between their actual GPU and mapped HOST
cells; MTP overlaps the bounded host/copy job and next target waits on publication.
No expert IDs/weights are substituted and no extra model copy is downloaded.

Sourcedfe2e6f7 adds only a resource aggregate: record swap count AFTER successful
copy join, private map/usage writes and publication. Only fixed-public completion
metadata exports it; no routing sequence, prompts or tensor values persist.
Frozen strata-adaptive-four-20261019, sourcedfe2e6f71ab4280efb216d687c4f3cddef4b457d,
SHA2564cd6d4f7478e1393321a32cf17540213888663be6f7862b2d7e594a90aac78b6,
features strata-rust-port,strata-intel-core,strata-gpu-adaptive-usage.
37 actual global cache/source/ownership unit cases pass with the feature enabled;
strict same-feature Clippy and401 shader/32 matrix validation pass. The earlier
130 operator cases still cover unchanged projection math, not a rerun here.

strata-adaptive-four-retained-20261019/intel.json passes two identical362-output/
EOS requests,108 windows/32 repairs in the second report, explicit window4,
gpu_adaptive_expert_cache=true and4887 FULLY PUBLISHED expert swaps over the
loaded Session. Thus the actual mutable source/cache/copy/map/rebind path ran.
Target/draft erasure, final close and root recovery pass: Intel50,688,000B,
NVIDIA compute0, normal admission/allHTTP200 and all driver faults0. Active
swap/core0 and original48.2GiB cap observed. Visible unsaved public tail coherent.
No cross-artifact token equality, independent physical byte audit of every swap,
private-caller, maximum-context, mixed-prompt or soak acceptance is implied.

PASS for exact fixed-public retained performance measurement of this candidate.
The manifest records window4 and requires enabled adaptive mode with nonzero
published swaps before a timing receipt may be written. Older frozen acceptances
remain nested historical controls. No adaptive speed result yet at this decision.


## Adaptive cache crosses thirty tokens per second (2026-10-18 local)

strata-adaptive-four-throughput-20261019/throughput.json passes exact accepted
dfe2e6f7/SHA4cd6d4f7, four-input native requests with adaptive usage enabled.
Two same-session identical362-output/EOS requests,108 windows/32 repairs in
second report;4887 fully published expert swaps over the Session.

| Interval | First request | Warm second request |
| --- | --- | --- |
| Prefix/preflight |5.460215250s |5.240704524s |
| Last-prompt-inclusive generation |15.107438066s |10.847832113s |
| Generation tokens/s |23.961706705 |33.370722946 |
| Full request before erasure |20.567653316s |16.088536637s |
| Full request tokens/s |17.600452246 |22.500492628 |

Target-phase runtime setup98.395355839s excludes earlier process/device/file
setup. These are delivered-token engine rates, not pure GPU decode or a new
Ollama comparison. Generation includes final prompt input and first prediction,
all target/MTP/controller/cache work, but excludes prefix/preflight, request
erasure and model close. This is one fixed-public code benchmark and one warm
sample, not a universal rate or statistical distribution. Different topics,
contexts and cache heat may differ. Prior controls remain21.3057 separate-GU,
20.9871 joint-GU,18.9600 eight-input; no partial coupled component was removed
because of those timing results. The normal four-input configuration remains.

Both actual request owners erase and final close completes. Root after-state:
Intel50,688,000B,NVIDIA compute0,normal admission/allHTTP200,all classified faults0.
Swap/core0 and original48.2GiB cap were observed active. The thirty-token warm
measurement milestone is met on the selected B70/original DASLab/Q2 configuration.
A final alternate public prose/cap check is in progress; no further optimization
is started in this block. No installed native backend/API/frontend/release change.


## Final selected-engine checkpoint (2026-10-18 local)

The selected connected Intel rewrite and >=30 warm delivered-token milestone
are complete for the documented fixed-public scope. No GPU test remains running.
The final additional correctness check strata-adaptive-four-prose-20261019
passes actual512 outputs/no EOS at its requested512 cap,185 windows/72 repairs,
window4,adaptive=true,3247 fully published swaps. The unsaved public tail is a
coherent hash-table explanation cut off by the requested cap. Target/draft
request erasure and final all-peer close pass. Root final after-state remains
Intel50,688,000B,NVIDIA compute0 (538MiB desktop), allHTTP200, admission active/
waiting/paused/recovery_required=false/0, all classified driver faults0. Active
swap/core0 was observed. No code-generation execution-quality claim is made.

Canonical performance build remains frozen dfe2e6f7/SHA4cd6d4f7, explicit
--intel-core --gpu-adaptive-usage; ordinary --intel-core alone does NOT enable
this experimental cache extension. --eight is optional and measured slower on
the same fixture; four remains default. Exact existing frozen artifact must be
used for reproducing the accepted timing manifest. A fresh runtime/source build
requires its own frozen identity and documented acceptance before advancing the
allowlist. Metadata-only docs/runner commits do not replace frozen runtime IDs.

Worktree /home/jmicko/.codex/worktrees/kuno-image-speed/vashti, branch
codex/strata-rust-port. Source commits this block: dcd7326e direct planes;
55294dda original-weight actual input parity;883d584b explicit product precision;
b91bcb33/0b76f8bb joint GU/coordinator and regular helper parameters;45a8eb6a
explicit eight-input request;90968c5a strict admitted head-counter capacity;
dfe2e6f7 credit only fully published exchanges in fixed-public metadata.
The two failed intermediate source/numerical trials and the eight-head accounting
failure remain failed; later receipts supersede them without rewriting evidence.

Next work is separately scoped production/native-service integration and an
independently protected arbitrary/private caller, plus mixed requests on one
retained Session, context extremes, broader stochastic behavior and soak.
The selected engine milestone is not proof of those exclusions or a generally
stable33-token service. Native runtime is still a development harness, not the
installed inference backend. No Vashti consumer/API/UI change is required for
these internal engine changes; agree shared contracts before integrating them.
No native install/release, model replacement, GPU reset/reboot or driver change.
One graceful installed Kuno restart early in the block only evicted its idle
image model, with services subsequently verified healthy. Original inherited
untracked benchmarks/text/__pycache__ remains preserved.
