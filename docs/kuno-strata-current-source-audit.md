# Current Strata source audit

Runtime source: `fad14f5dba7395a66cc2c06aea649fe1feeb1ca6`, isolated `codex/strata-rust-port`.
Pinned Intel upstream: `fb58e0dbc8399662c0e47c76578c6e878b14f6cf`.
This is the current audit index, alongside the historical evidence in
[kuno-strata-rewrite-completion.md](kuno-strata-rewrite-completion.md).
**Selected engine source-wiring audit passed; whole rewrite acceptance remains
open.** See the explicit decision and scope below. Physical correctness,
resource recovery and performance are separate gates. Work remains CPU-only
pending renewed B70 authorization; this document grants no GPU/service/install/
release authorization.

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

## Remaining acceptance work

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
