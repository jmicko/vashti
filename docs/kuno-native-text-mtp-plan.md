# Native Unsloth quant and Qwen4Exp MTP

The user approved the Unsloth UD-Q3_K_XL main model and self-contained Q8_0 MTP
head, and explicitly chose our Rust implementation. No replacement C/C++ text
engine is being introduced. This is internal laboratory work; production routing
and Vashti consumers are unchanged.
The current approved main model is DASLab IQ3_S; the Unsloth main-model trials
below are historical. The original self-contained Q8 MTP sidecar remains useful.

## Backend direction approved by the user

Kuno should prefer a supported native Rust backend and use Ollama when it has no
supported native implementation. Support must cover the actual model artifact,
geometry, quant formats and requested inference behavior, not just its model
name. Choose the backend before execution. A native runtime failure after work
has begun is not permission to replay silently in Ollama. Keep an explicit
Ollama selection for comparisons. Its exact naming/API contract remains future
shared-contract work; this milestone adds no routing or catalog override.

## Pinned artifacts and storage

Publisher: `unsloth/Qwen3.8-Flash-Next-GGUF`, revision
`766911a6b7369840a91dbcd95f9f997acaab6cd6`.

- UD-Q3_K_XL is three original GGUF shards totaling89,986,353,824bytes.
  Shard0 is metadata-only; shards1/2 contain402/822 tensors,1224 total.
- Its actual formats are F32(557), BF16(24), Q8_0(502), Q6_K(1),
  IQ3_XXS(94), IQ4_NL(44), IQ4_XS(2). Nominal Q3 does not imply uniform packing.
- The self-contained Q8 MTP head is4,137,429,120bytes with34 tensors:
  Q8_0(21), F32(11), BF16(2). It stores its own embedding and output head,
  simplifying a future independent RTX3060 owner.
- Verify complete SHA256 values before import or inference. Keep one payload on
  NVMe: Ollama's existing blob store plus same-filesystem named hardlinks for
  split views. An unreferenced MTP sidecar must not rely on Ollama's blob GC.
- Ollama0.35.1 supports local multi-shard `create`, despite the remote registry
  rejecting shard pulls. Its pinned b11232 backend lacks this MTP graph. The
  ordinary model remains a possible comparison; a DRAFT import alone would not
  establish supported MTP execution.

**Download and import are complete.** All four full-file SHA256 values match the
pinned publisher. The three main payloads are hardlinked into Ollama's CAS and
registered as `qwen3.8-flash-next:125b-a6b-ud-q3_k_xl`; no merge or second payload
copy was made. The Q8 sidecar has a stable named hardlink outside Ollama's GC
namespace and is not registered as an unsupported Ollama draft.

The public download receipt and bounded tensor inventories live under
`/media/jmicko/faststore/kuno-development` and the model import directory. They
contain public weight provenance only. No private content or expert histories
belong in them. Exact provenance and acceptance are recorded in the
[foundation receipt](../apps/kuno/benchmarks/text/receipts/unsloth-q3-foundation-20261008.json).

## First native MTP reference

`crates/kuno-text/src/mtp.rs` binds only the inspected self-contained Q8 head.
It independently translates the [pinned upstream graph](https://github.com/danielhanchen/llama.cpp/blob/6fcaa16f4b360649933a54d1f91ad40ed35c0e11/src/models/qwen4exp.cpp)
into Rust; no upstream engine is linked. Exact tensor names, dimensions and
types are whitelisted. Metadata requires49 blocks, one NextN block, four
2560-wide HC streams, LR320,512 experts/top10/FF640,24 query/2 KV heads,
256-wide heads,64 rotary dimensions, theta1e7 and RMS epsilon1e-6. All34 stored
tensors are checked, including four unused indexer tensors.

For each token, normalize the preceding10240-wide residual as one global RMS,
normalize its2560-wide embedding, concatenate embedding then that stream's
hidden state, and apply the same5120→2560 EH projection four times. Perform
HC mixer → dense causal GQA → HC combine → HC mixer → routed and shared MoE →
HC combine. Return the resulting wide residual before the final HC head mixer
and248320-wide output projection. Attention Q and gate are interleaved per head.
Block48 uses dense attention, independent of ordinary trunk modulo scheduling
and the stored QSA indexer. Reference context is explicitly limited to2048.

Each sequence owns its KV cache and binding identity. A rejected preflight input
leaves state unchanged. An execution error clears/poisons it until reset. Draft
rewind clears the provisional KV suffix; target GDN/PLE rollback is a separate
unimplemented transaction. Returned logits/hidden and MTP-owned intermediate
buffers use wiping guards. Existing scalar normalization/HC helpers have not
received whole-stack error-path erasure acceptance; this reference remains a
public-fixture laboratory path, never a live HTTP/private-request engine.

Unit tests cover embedding-first per-stream conditioning, per-head Q/gate
ordering, independent dense-attention equations,4-query/2-KV grouping, all
visible positions past2048, cancellation, and observed cache wiping helpers.
An opt-in downloaded-head test checks strict real artifact binding, complete
graph scheduling with a synthetic executor, cross-binding rejection, failure
after KV publication, reset and fresh replay. It proves neither full-head
numerical parity nor useful speculative performance.

### Publisher conditioning correction, 2026-10-09

`724ca2f` corrects only input hidden normalization from four2560-wide RMS groups
to one10240-wide RMS before the shared EH projection. Attention/FFN/head HC
mixers retain their original per-stream norms. The analytic host test preserves
different amplitudes across all four streams; full core117 tests pass (two
opt-in ignored), as do strict library Clippy and formatting. GPU acceptance of
the corrected resident MTP workspace and a new full-vector oracle is pending.

The original pinned llama prototype differs from the publisher here. Official
[vLLM0.30 NVIDIA MTP](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/models/qwen4_exp/nvidia/mtp.py)
and [AMD MTP](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/models/qwen4_exp/amd/mtp.py)
flatten all four HC streams for `pre_fc_norm_hidden`. Its
[base proposer](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/v1/spec_decode/llm_base_proposer.py)
shifts token IDs while retaining target positions and hidden states. Thus MTP
cell c consumes token T(c+1) with target residual R(c), at RoPE position c,
and predicts T(c+2). A next provisional cell consumes the draft token and the
draft's own residual. Strata's
[pinned cell contract](https://github.com/Niko1221/Strata/blob/fb58e0dbc8399662c0e47c76578c6e878b14f6cf/include/strata/core/mtp.hpp)
independently follows this convention. Normal coupled inference starts from
real target R0 and T1; it does not insert a dummy T0/zero-hidden KV entry.

Historical zero-hidden-first/per-stream fixtures below remain exact records of
cross-executor repeatability under the old convention. They do not establish
publisher-compatible conditioning or draft quality. The preserved six-row CPU
fixture at `0c1cbfb` is likewise historical: its first two provisional greedies
miss the DAS target. Do not reuse that binary reference for the corrected
global-norm/aligned-cell graph. The replacement header must explicitly version
both conditioning and math before numerical or speculative speed acceptance.

## Implemented foundation, 2026-10-08

- Strict multi-shard ownership pins each original source and validates the
  complete split inventory. Original-packed IQ3_XXS, IQ4_NL and IQ4_XS scalar
  decoders match 8,032 independently generated upstream C values bit-for-bit.
  The C reference is test data; no C/C++ text engine is linked.
- Rust GPU IQ decode/projection checks pass on B70 and RTX3060: 16,064 exact
  decoded values including unaligned prefixes, 117 projection/MAX8 cases,
  12 cooperative finite/fault cases and 12 subgroup cases, with full private
  owning-buffer and public-source checks. The IQ4 scalar-literal optimization
  required explicit assignments: CubeCL incorrectly lowered a nested value
  expression to the last lookup word. A generated-shader regression check and
  both physical devices accept the corrected `4b29087` source.
- The actual 1,224-tensor smaller main graph binds through its explicit
  `UnslothQ3` profile, including its original 90-byte IQ4_NL PLE rows. Generic
  grouped projections preserve mixed types; Q4-only shortcuts remain rejected.
  The fixed `!` numerical diagnostic checks all 248,320 logits and exact greedy
  IDs at each of four or 32 positions against the serial-row CPU graph. This
  is cross-executor parity, not an independent whole-model quality oracle.
- Q3's 144-slot banks charge 19,998,441,472 bytes. Ordinary projection pages
  charge 5.375 GiB, exceeding the former 5 GiB cache. The explicit 8 GiB profile
  removes repeated ordinary-weight uploads while retaining the same 31 GiB
  device budget and checked margin. Original Q4 budgets remain unchanged.
- Direct packed IQ CPU AVX2 math retains the original 64-lane separate
  multiply/add order, including the IQ path under the explicit FMA backend.
  Seven paired public microbenchmarks show 6.82/10.57/12.15x arithmetic gains
  for IQ3_XXS/IQ4_XS/IQ4_NL. These are not whole-model throughput claims.
- `Qwen4Exp::step_with_carry` returns guarded logits and the actual 10,240-wide
  residual before the final HC mixer, with consumed position and binding ID.
  The legacy `step` shares one original head and moves its logits allocation.
  The protected real-weight CPU coupling test passes two target executions
  and four MTP executions, using real target h0 at draft position1. Full
  target logits and both draft hidden/logit vectors have zero fresh-replay bit
  mismatches. Target and MTP
  binding domains are checked separately.
- Protected default-build `mtp-gpu-probe` at `8cba745` passes on both physical
  cards. It consumes fixed token0/zero preceding hidden at position0 and the
  real target-greedy token/target h0 at position1. Every248320-wide draft logit
  vector and10240-wide hidden vector matches the CPU graph within the unchanged
  explicit tolerance, with exact greedy IDs and zero fresh GPU replay bit
  mismatches. Each two-position sequence executes132 original-packed GPU
  projection kernels. Main allocator allocations/in-use/padding/reserved bytes
  return to their zero baseline after checked synchronization and cleanup.
  Other MTP graph operations execute on CPU. Separate staging/uniform owning
  erasure, an independent head oracle and performance remain unaccepted.

The compiler-idle32-position Q3 run at `4b29087` passes every full-logit and
greedy check, all original owning closes and fresh/recovery checks. Its31 timed
decode continuations take4250ms: **7.294tokens/s**. Ordinary weight uploads are
zero; observed physical reads are1,323,008bytes. GPU experts cover1940 of14880
decode selections;12940 still execute on CPU. One-token prefill seeds only ten
of144 slots per layer and the current static CPU-miss policy leaves the rest
empty. Explicit empty-slot filling at `b739c32` is implemented but not accepted:
its unchanged32-position test reproduces the same finite position23 mismatch
as the GPU-miss control below, with exactly the same first failing value and
1.176437 maximum logit error. The no-eviction proof excludes replacement as
the cause. The candidate remains opt-in. The completed same-input expert/router diagnosis
below distinguishes projection arithmetic from discrete routing amplification.

### Routing diagnosis and bounded performance comparison

The fixed public same-input shadow at `37f4edb` compares all48 layers at
position23:1,228,800 expert output values, maximum absolute difference
1.7881393e-7. The router comparison at `67bdcfa` locates the first selected-set
change at layer17. CPU's tenth/eleventh score gap is1.001358e-5; the GPU's
scores tie exactly, with maximum score difference2.384186e-5. Both use their
original tie rule. Applying CPU top-k to the GPU's actual scores reproduces
all ten selected experts and ranks at all48 layers. The next layer's score
error grows to0.023154; the final strict logit gate still fails. This is a
routing discontinuity following small arithmetic differences, not evidence of
an expert decoder or cache eviction defect. No wrong-expert substitution,
acceptance threshold change or full32 timing acceptance was introduced.

For a bounded cache comparison, the explicit23-position mode checks the first
23 complete logit vectors of the original32-position CPU file. The complete
32-position file and unused suffix must still parse correctly; no shortened
reference is generated. There are22 timed continuations, ending before the
known position23 discontinuity. Both strict4/32 modes and the full32 failure
are retained. Prefix receipts explicitly identify this cutoff and cannot claim
full32 reference acceptance. Matched v88 static/empty-fill runs pass complete
original owning/recovery checks:22 decode steps take2995ms/4188ms, respectively
(7.346/5.253tokens/s). Demand-fill uploads2813 experts, filling480→3293 of6912
slots; its CPU rank count falls tozero, but filling is slower on this sample.
There is no demand-cache speedup claim.

A fresh correct-expert ideal replay at `9218638` records the actual22×48×10
selections into an erased transient public-only trace, closes the training owner,
then preloads their union into a fresh original owner. The per-layer union must
fit144 slots, and every actual replay selection must belong to it. This does
not force expert choices or reset a poisoned ledger. The3102-expert union fits
38–135/layer;6459852800 original preload bytes are excluded from timing.
Both replay runs verify all23 complete original logit vectors and greedy IDs,
10,560 GPU hits, zero CPU expert calls/weight uploads, and full owning closes.
Same frozen v90 binary gives2311859249ns (9.516tokens/s) without the explicit
unused-segment option and2174240700ns (10.118tokens/s) with it: one ordered pair,
5.95% lower decode latency, not statistical proof. Both timed windows observe
zero physical reads, but17/47 major faults; zero I/O does not mean zero faults.
Timer includes full-vector/trace checks and audited request-I/O close; it
excludes training/binding/prefill/preload and later full close/recovery.

The skip candidate (`397999c`) derives a segment-activity mask only from the
validated immutable pending GPU-slot plan. Original Q3 gate/up and down banks
are paired72+72 segments. A wholly unused segment omits only its two projection
launches; original selected rank order, partial spans, finite/status checks,
physical/source/pin proofs and merge remain unchanged. Default dispatch remains
unchanged. v90 initial fill without/with skip takes5401/4141ms, but physical
reads differ605462528/49152bytes; this is not an isolated kernel speedup.
Keep both empty-fill and skip opt-in until longer correctness/performance work.

Root source `9218638` passes591 native plus two one-test shader suites, strict
release all-target default/all-five-feature Clippy and formatting. Four physical
v90 runs pass five original owning audits each; ideal training/replay adds two
full independent closes. Host admission/services and physical memory recover,
with no new classified Xe/NVIDIA fault. No production/installation/API change.

The user paused this work to use the GPUs. Isolated leaf candidate `6d12ddd`
(`codex/kuno-q3-subgroup-moe`, base `9218638`) uses original `fast::dot` and
`reduce_original64`, four rows per128-thread group. Source/formatting only:
no compilation, count1 caller, independent probe extension or GPU acceptance.
Resume by connecting explicit protected opt-in/device subgroup support and
extending the existing three mixed-IQ C-oracle profiles, fault/status/owning
checks, then host/both-card/operator/full-vector performance tests. It is not
integrated into this root checkpoint.

The same frozen artifact's all-GPU-miss32-position control passes positions0–22
but fails the full-logit check at position23, after greedy equality passes.
There is no accepted performance receipt for that control. Earlier maximum
errors stay below0.00009, far below the0.006 absolute tolerance floor; the
failure must be diagnosed rather than described as ordinary accumulated drift.
Acceptance tolerances are unchanged and normal admission/GPU memory recover.

The first result comes from prefill: four outputs represent three timed decode
continuations, and 32 outputs represent 31. Report that count. The diagnostic's
generation timer includes comparisons, logging, counters and request-I/O close
auditing; it excludes model binding, prefill and later recovery tests. Mapped
CPU projection stages include scheduling and page faults. They are not isolated
compute/read timings. Short-run cache, placement and SIMD results also differ
in physical I/O; the GPU-miss control disables request/binding reuse. Preserve
these limits when interpreting measurements.

MTP's CPU reference and any nonresident GPU projection smoke remain fixed-public
laboratory paths. Inherited scalar intermediates and the legacy nonresident
GPU path lack whole-stack private owning-erasure acceptance. They are not
eligible for private requests or native HTTP routing. No speculative speedup
has been established.

## Following implementation and acceptance

1. Preserve the accepted split, original packed operators and explicit Q3
   ownership above. Extend quality/context and meaningful throughput coverage;
   different quants cannot reuse Q4 token IDs as a quality oracle.
2. Validate complete MTP head numerical output against an independent graph
   oracle. CPU fresh replay proves repeatability, not correct full-head math.
   The two-position GPU projection check against the CPU graph and real target
   carry now passes both cards; keep this legacy nonresident probe public-only.
3. Extend the implemented CPU target carry to the GPU owner and return logits
   and wide hidden for every verification position. Existing MAX8 LastLogits
   only projects the last row.
4. Start with at most two drafts: confirmed token plus0–2 drafts uses a1–3-row
   target verification. Commit only the accepted prefix and produce the target
   correction/bonus token. Greedy equality first; exact sampled rejection is a
   separate milestone. Never substitute resident experts to manufacture speed.
5. Restore the latest pre-proposal GDN recurrent **and raw convolution** roots,
   clear only the appended QSA suffix, and restore PLE history/cursors before
   recomputing an accepted prefix when a draft is rejected. The original double
   state owners retain the prior GDN roots until another chunk. Bounded latest
   tickets cost1MiB host/49,152bytes INFO with no extra GPU state; exclusive
   round coordination and physical repair acceptance remain pending. Merely
   changing sequence.position is wrong. Extra per-prefix recurrent snapshots
   would cost216MiB logical memory before other families and are unnecessary
   for this initial rollback/recompute strategy.
6. Repair draft KV with real target hidden states after acceptance. Pair token
   T(c+1) with target R(c) at MTP cell c, not at target token position c+1.
   A provisional draft's own hidden state
   must not remain authoritative in accepted catch-up history.
7. Build dedicated dense-attention and all-resident Q8 draft owners, explicit
   expert-buffer sharding and cross-device bridges. The existing main/secondary
   slot limits do not authorize a512-expert resident draft by changing a number.
8. Compare the same public fixture with MTP off, same-device draft, and RTX3060
   draft/B70 target. Include load/prefill, accepted-token throughput, acceptance
   rate, bridge/repair costs, context, physical disk I/O and GPU recovery. Do not
   present draft tokens/s as accepted response throughput.

See the [pinned speculative driver](https://github.com/danielhanchen/llama.cpp/blob/6fcaa16f4b360649933a54d1f91ad40ed35c0e11/common/speculative.cpp),
[rollback fix](https://github.com/ggml-org/llama.cpp/pull/28123), and
[publisher's MTP guide](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/blob/766911a6b7369840a91dbcd95f9f997acaab6cd6/MTP/README.md).
Published speedups are not local acceptance. No GPU reset, driver/clock change,
installed binary/service change, release, API change, or Vashti work is included.

### DASLab comparison candidate, 2026-10-08

The user asked to revisit DASLab's quality after reviewing Strata. The current
[publisher card](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF)
describes GSQ weight optimization plus RCO per-tensor precision allocation
under a size budget. Its83.6GB IQ3_S results are close to the BF16 base on
the listed reasoning/coding tasks; this is a publisher result, not a direct
Unsloth comparison or a measurement on this host. The earlier pinned artifact
assessment in the disk research remains historical. Three former missing
formats and split/PLE ownership are now implemented here. IQ2_S, IQ3_S,
Q2_0 and Q5_K still need actual inventory-driven support before that assessed
IQ3_S artifact can run. No DASLab payload was downloaded in this block; the
current Unsloth work and its original public references remain intact.
