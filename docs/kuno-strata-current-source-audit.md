# Current Strata source audit

Runtime source: `9a3567e6afe57de88835645786c72a0012b86200`, isolated `codex/strata-rust-port`.
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

## GR/GDN selected defaults: call sites traced further

Actual `resident_model_batch::run_global` consumes pending GR writes, PLE at
layer1, GR attention read, prepared GDN/QSA, GR FFN read, actual routed/shared
fork/join and final GR/head. Pinned `fused_gr.dp.cpp` selects `gr_norm_split()`
and `gr_down_sliced()` by default: split256-thread stream normalization,
80 slices, ordered slice reduction, then up/read. Rust's actual prepared roots
and `strata_gr` dispatches select that family. The newer hc checks/variants in
the same upstream file are not this default: absent `STRATA_HC_CHECK`, its actual
selector is kHcPlain. This traces the selected family; remaining numeric details
and physical source/layout acceptance still belong to the full operator audit.

Pinned `verify.cpp` invokes multi conv/L2, BF16 alpha/beta, native quantized Z and
`gdn_step_norm_multi`. Its default in `verify_kernels.dp.cpp` is the512-thread
four-row-group/128-column step; `STRATA_GDN_SPLIT` is opt-in. Rust's actual
`resident_gdn_batch` uses that512-thread recurrence, original deferred raw history
and state pair; native128-thread accepted-state commit and deferred prefix state
remain as documented earlier. Do not mistake that separate commit default for a
missing split-step optimization or infer defaults from the CUDA sibling.

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

## Concrete remaining PLE reader gap: default cache and asynchronous transport

Allowlisted extraction from the saved reference argv records `--ple-io direct`,
with no `--ple-row-cache`, `--ple-inflight` or `--ple-sync-submit` override.
`strata-ple-reference-options-20261017.json` contains only those public fields.
Pinned Intel `generate.cpp:429..434,2926..2929` therefore selects its original
1,048,576-row cache,256 inflight jobs and I/O worker; the default SSD keepalive
is100ms for60 seconds after row requests. `ngram/ple_reader.cpp::RowCache`
uses eight ways/set, original64-bit wrapping mix, round-robin replacement and
packed original bytes. The actual target's table is IQ4_NL `[160,320001536]`,
90B/row. Cache data/keys/replacement owners alone total98,697,216B (94.125MiB),
plus metadata. This is a derived source inventory, not runtime measurement.

Current `resident_ple_batch::native_embedding` has a stable sorted/unique plan,
four joined positional workers, original decoder and destination scatter. Prompt
already double-buffers next-chunk source work. Neither path has the native bounded
row cache or original256-job direct-I/O engine. Deduplication within one batch is
not reuse across batches. These are genuine remaining selected-source differences,
not optional tuning or proof that the whole reader has been ported. Existing
privacy statements forbidding persistence do not prohibit a wiped request cache.

Next implementation must connect the reader as a whole:

- One original bounded packed-row cache shared by decode and actual prompt
  prefetch/fallback, authenticated to the original model/table binding. Preserve
  eight-way replacement and real hit/miss reads; never return stale/cross-model
  data or silently treat fresh rows as cached. Inspect the original page grouping,
  aligned direct reads, inflight queue/completions and actual gather issuance/join
  points before choosing a Rust transport. Document necessary backend substitutions.
- Explicit request lifetime/worker ownership, including error/cancellation/unwind,
  original descriptor/stat checks and joins before source closure. Kuno must wipe
  all allocated row IDs, cached bytes, replacement state, queued plans and private
  staging before request completion/retained rebinding. No row cache/prompt/content
  persists or enters a log. A cache retained across requests would violate this
  privacy boundary; rebuilding it is a necessary lifetime substitution.
- Charge actual cache/queue/aligned buffer/worker/metadata capacity in the original
  HOST constructor/reservation inventory (including peak coexistence with prompt
  sources and cleanup). It is not covered by merely enlarging the old4MiB reader
  cap or borrowing uncharged GPU MAIN. Preserve original no-swap/core protections.
- Prove original packed hit/miss bytes, collisions/replacement/duplicate rows,
  page-straddling/tails, prompt/decode shared use, original identity/stat refusal,
  bounded admission, cancellation/error/drain and actual erasure with public CPU
  fixtures, then normal build/offline/full-engine correctness gates. Do not use
  a microbenchmark or interim speed result to replace the complete reader port.

## Source review still to finish

- Consolidate actual selected dense/expert matrix and quantized arithmetic
  producers, GR/GDN/QSA/PLE operators and prompt overlays against the pinned
  call sites and prior receipts. Explicitly accept or resolve every remaining
  storage, grouping, queue and math substitution; filenames alone do not pass.
- Preserve the now-traced per-request126/128 graph lifetime and original
  owner erasure. Physical retained mixed-request execution still needs acceptance.
- Port the actual default PLE row cache/asynchronous reader described above,
  including its connected decode/prompt request ownership and full HOST pricing.
- Review remaining queue/INFO/descriptor hazards, shared/MTP fork/commit order,
  sampler state and output/resource closure as one connected schedule.

Then, with renewed GPU authorization and supported protection, require whole-
model prompt/count1/multirow, acceptance/rejection, EOS/cap, context, retained/
hot-request and failure/all-resource recovery acceptance. Only after source
completeness and those correctness gates pass may live speed measurements begin.
