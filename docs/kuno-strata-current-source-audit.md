# Current Strata source audit

Runtime source: `1b8d354d56c501cda08ec6404ce2f5427732e507`, isolated `codex/strata-rust-port`.
Pinned Intel upstream: `fb58e0dbc8399662c0e47c76578c6e878b14f6cf`.
This is the current audit index, alongside the historical evidence in
[kuno-strata-rewrite-completion.md](kuno-strata-rewrite-completion.md).
**Whole-source completeness is not yet accepted.** CPU/source review, physical
correctness, resource recovery and performance are separate gates. Work remains
CPU-only; this document grants no GPU/service/install/release authorization.

## Entry, loading and cleanup: call chain traced

The selected executable's actual `daslab_language_probe.rs` entry constructs the
owned request and calls `strata_engine::execute`. `Session::load/load_inner`
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
pretending completion. **Actual driver residency, no-swap protection, repeat
request execution and full GPU recovery remain unverified.**

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

## Source review still to finish

- Consolidate selected quantized arithmetic and remaining prompt dense/expert
  overlays against the pinned
  call sites and prior receipts. GR/GDN/QSA decode/prompt consumer review is
  indexed above. Dense/ranked source/loan ownership is traced above; physical
  placement/numeric/state acceptance is still open. Explicitly
  accept or resolve every remaining storage, grouping, queue and math
  substitution; filenames alone do not pass.
- Preserve the now-traced per-request126/128 graph lifetime and original
  owner erasure. Physical retained mixed-request execution still needs acceptance.
- Validate the connected two-program PLE handoff above on the real protected
  device, including cold/capture/replay and failure joins; source connection
  and CPU parameter/reader coverage are not physical fence/overlap proof.
- Review remaining queue/INFO/descriptor hazards, shared/MTP fork/commit order,
  sampler state and output/resource closure as one connected schedule.

Then, with renewed GPU authorization and supported protection, require whole-
model prompt/count1/multirow, acceptance/rejection, EOS/cap, context, retained/
hot-request and failure/all-resource recovery acceptance. Only after source
completeness and those correctness gates pass may live speed measurements begin.
