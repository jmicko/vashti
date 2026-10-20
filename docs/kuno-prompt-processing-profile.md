# Native prompt processing investigation — 2026-10-19

**Superseded by installed continuation (2026-10-20):**
[Strata gap investigation](kuno-prefill-strata-gap-investigation.md) records the
selected router/vector-dequant/K64 paired-buffer matrix plus bounded short-prefix
scope pathc8db19de / SHAfb31e124. Clean matched4245 prefix43.2055 ->24.9658s,
42.2% shorter; staged35-token prefix1.290s. All exact numerical and private staged
thinking/tool/semantic/privacy/recovery gates pass. Official install preserves
the original gateway/configuration; final installed public/metadata checks pass.
Installed warm35-token prefix1.4305s (old4.639s); code ends normally at501tokens
and passes155 merge cases twice with identical replies. Zero worker writes/OOM/
driver faults, swap/core0, normal admission and all services200; weights stay warm.
No task test client/staged service or sudo-refresh loop remains.
The source60c20e36 results below are the historical baseline for this continuation.

This is Kuno backend work in `codex/strata-rust-port`. No Vashti source or API
change is involved. The accepted direct-buffer worker below is installed through the unchanged
image-enabled gateway. Final installed request/metadata acceptance passes.

## Workload and measurement

The dedicated `--public-prefill-{profile,check}` entry accepts a model-settings
path, never stdin or caller text. It runs three compiled public requests on one
Session: 45 input tokens, then the same 4,245-token input twice. The longer
input adds 300 repetitions of public reference material before the same arithmetic
question (17 times 19). Temperature0, seed0, context32768, output cap32; each
completed request returns four tokens containing323. Reply bytes remain in RAM.
Actual request erasure and Session close are required before a passed receipt.

`--public-prefill-profile` adds completed GPU fences at coarse operator boundaries.
Its times are diagnostic, NOT ordinary throughput. The static per-category tuple
is `(calls, elapsed_ns, enqueue_elapsed_ns)`; the first baseline predates the
third field. In the uninstalled5fbc3ca5 candidate, labels `expert_group_count`,
`expert_fp16_image_bytes`, `expert_host_stage_bytes` and
`expert_host_copy_segments` instead put aggregate counts/bytes in slot0 and zero
in both time slots. These counters are compiled-public-only and contain no route
values, source IDs or contents. Enqueue time includes any blocking inside the called backend, not
just CPU arithmetic. Nested attention categories overlap their outer category
and must not be summed twice. `expert_cpu_*` slices do not add device fences.
The private worker cannot activate any profiling through input or environment.

The reported prompt interval ends before the final count1 input, as in the
installed worker. Loading, generated output and completed-request erasure are
separate. Neither prompt-cache retention nor the privacy contract is changed.

Receipts/artifacts below are under `/media/jmicko/faststore/kuno-development`.
Every protected runner disables swap/core dumps, unloads the native worker,
pauses gateway admission, uses the real B70, then requires graceful close,
restored physical memory, normal admission and unchanged driver-fault counts.
Only static/public metadata and numeric results are saved.

## Measured baseline

- Source `342970e2`, artifact `strata-prefill-profile-baseline-20261019`, SHA
  `62c9f917eb8b75c8a52ab8fc4dc21a27faa7942358ef8e10c8ed0209d79ca5ff`.
  `prefill-profile-baseline-20261019/receipt.json`: warm repeated prompt56.596s,
  experts35.532s, attention15.934s, MTP0.187s. All three replies/correctness,
  request erasure/close and physical recovery pass.
- Source `4a33422e`, artifact `strata-prefill-member-grids-20261019`, SHA
  `df9a77bffbd1604975a0a9999d394f16db029ab37e7408b5bed1a0e49c40bc94`.
  `prefill-member-profile-20261019/receipt.json`: warm54.998s, experts33.966s,
  attention15.926s. Bounding gather/activation grids to the actual expert's
  routed members saves about1.6s in this diagnostic, but is not the main fix.
- Source `f9471ce8`, artifact `strata-prefill-attention-profile-20261019`, SHA
  `6388885a8bd7b6007b08d00911330ea593fed7b3782e30a556830f9369974cfb`.
  `prefill-attention-profile-20261019/receipt.json`: warm55.231s. GDN input
  projections6.637s/output2.543s; QSA input1.958s/output0.852s; QSA attention
  3.006s. GDN recurrence/norm only0.892s, so register-state/recurrence changes
  are not selected. This diagnostic also passes complete recovery.

## Selected changes and validation scope

- `342970e2`: compiled-public profile/check callers and fenced coarse timing.
- `4a33422e`: expert member-bound gather/activation grids, keeping the full
  original descriptor and shader ownership bounds; adds grid coverage checks.
- `f9471ce8`: attention projection/recurrence subdivisions, diagnostic only.
- `ef886df5`: queue original same-stream expert kernels/copies without an
  intermediate scheduler flush per expert. The native graph retains original
  physical read/write intervals, ring release dependencies, quotas, error scopes
  and final actual completion/fault checks. Ordinary non-native scheduling still
  flushes. Adds non-submitting original-order scheduler tests. Also stages K64
  shared matrix operands, issuing the same four ordered K16 cooperative products
  between barriers. FP16/BF16 rounding and accumulation type remain unchanged.
  The dedicated compiled-public matrix check uses the existing independent IEEE
  oracle, original model weights, full-owner/tail checks and erasure audit for
  counts1/8/35/257 in both formats (eight cases); no looser FTZ oracle is adopted.

CPU status at candidate preparation:27 prompt tests and strict selected worker
Clippy pass. Four scheduler tests pass by compiling the ORIGINAL vendored
scheduler source with a tiny Rust harness and the lab's actual dependencies.
Direct `cargo test -p cubecl-runtime` cannot run non-workspace dependency dev tests;
no vendor manifest/workspace membership was changed to work around that limit.
Harness: `prefill-scheduler-test-20261019.rs`; log
`prefill-scheduler-direct-final-tests-20261019.log`.

## Acceptance and installation

The selected direct-buffer candidate passes physical numerical, whole-model,
original-output, uninstrumented before/after and recovery checks below. Protected
private API/tool/streaming acceptance and the official installation transaction
now pass. Final installed request/metadata acceptance passes below.

### K64 control rejected; direct-buffer matrix candidate

`prefill-baseline-clean-20261019/receipt.json` supplies the uninstrumented control:
45-token prompt7.630s; repeated4,245-token prompt55.794/55.821s. Correctness,
erasure/close and physical recovery pass. These are this fixed public workload,
not the user's private four-word messages.

`ef886df5` / frozen SHA81c6294b passes all eight original-weight matrix IEEE cases
with unchanged tolerance, maximum absolute error4.2915344e-6; all actual owner/tail
and erasure checks pass (`prefill-batched-k64-matrix-20261019`). However its
profiled warm prompt is63.961s, so this coupled candidate is REJECTED for install.
The scheduling repair reduces the expert enqueue interval from33.081s to2.295s;
actual expert completion still takes34.906s. Larger shared-memory matrix tiles
also increase the attention interval from16.143s to21.736s. This is successful
numerical validation, not a performance improvement. Retain the control receipts.

The next candidate (`23454d39`, scalar-stride correction `60c20e36`) keeps the
scheduler repair but replaces K64 staging with direct cooperative loads from the
original GPU input/weight buffers for complete8x16/K16 tiles. All partial tiles
retain the original masked shared-memory path. Ordered K16 products, formats,
FP32 accumulation, original bounds and denormal handling stay unchanged. The
first build at23454d39 failed a GPU scalar stride type check and has NO frozen
artifact; the corrected selected worker passes strict Clippy. Physical numerical
and whole-model acceptance now pass for this direct-buffer candidate.
The additional original-scheduler cross-stream test passes (four tests total;
`prefill-scheduler-direct-final-tests-20261019.log`), proving a materialized
producer still submits before a cross-stream consumer is registered/executed.

### Selected direct-buffer result

Frozen source `60c20e36054fd2d00f610c93e0a07a557de1125a`, artifact
`strata-prefill-global-matrix-fixed-20261019`, worker SHA256
`1db4664d9530df7b079aea0c23c6e26a74fc450598784e6ab68388e62032cc8a`.
`prefill-global-matrix-numeric-20261019/receipt.json` passes all eight original
weight FP16/BF16 cases, zero mismatches against the unchanged IEEE oracle,
maximum absolute error4.2915344e-6, complete owner/tail checks and actual close.

The uninstrumented `prefill-global-matrix-clean-20261019/receipt.json` passes all
three public requests with the original correct four-token answer: 45-token
prompt6.780s, repeated4,245-token prompt43.018/43.205s. The matched warm repeat
falls from55.821 to43.205s: **22.6% shorter**, approximately1.29x. Cold load101.923s
is separate. This is one fixed public workload, not a measured improvement to
any particular private chat or a universal prompt rate. The longer prompt and
repeated request do not share content state: erasure remains real after each.

Fenced profile `prefill-global-matrix-profile-20261019` reports warm prompt43.465s,
experts25.761s, attention13.430s. The largest remaining completed interval is
expert work. The direct-buffer change also improves dense attention projections;
GDN recurrence alone remains a small part. Further prompt caching is a separate
proposal, below, not a shortcut in these measurements.

Offline validation passes401 ordinary shaders and32 cooperative matrix
configurations with the existing matrix-capable validator at
`spirv-tools-matrix-20261014/build/tools/spirv-val`
(`prefill-selected-matrix-validator-20261019.log`). The distribution validator
rejects BF16 capability5116; its failed run is retained separately and is not
reported as an engine validation pass.

A final eight-subgroup16x64 shared-operand candidate `d06c03d4` / SHA14f37aae
passes CPU/SPIRV status-plane tests but fails the protected small matrix numeric
mode at count35 after passing count1/8. It was stopped BEFORE whole-model or
speed testing and is not installed. Receipt/log: `prefill-block-matrix-numeric-20261019`
and `prefill-block-candidate-20261019.log`. Physical recovery passes with zero
driver faults. Forward selection commit `a37db7da` restores the complete matrix
source exactly to60c20e36, retaining the other validated changes. Failed candidate
receipts remain available; no oracle/bounds/fault checks were relaxed.

### Protected API and installation

`native-prefill-stage-check-20261019/{receipt,host-after}.json` passes the real
private worker through staged Kuno: capabilities, preload, ten complete public
tool schemas, streamed thinking/weather call, role=tool result/final answer on
the same PID, identical repeated362-token code replies, zero write_bytes, denied
unprivileged inspection, graceful unload and subsequent ordinary Ollama inference.
No real user notes/images/chats are read or written. Physical memory returns to
Intel50,688,000B, NVIDIA compute0, all services200, normal admission and unchanged
zero driver-fault counts.

The public toolbox first prompt has4,207 tokens and takes46.991s to prefill;
its4,293-token tool-result follow-up takes49.630s. This workload differs from the
repetitive4,245-token timing control and has no matched old-toolbox speed baseline
in this pass. Do not claim all chats are now22.6% faster or substitute this fixture
for the user's reported44s. Two35-token code prompts take4.598/4.647s to prefill;
the second362-token decode takes10.743s (33.70 output tokens/s). Both code replies
are identical; decoding and prefix computation are separate metrics.

The official `configure-native-text` transaction installs worker SHA1db4664d using
the unchanged image-enabled gateway SHA
`d3ed07f473b28bbb46596a619349d933edd6c166b7af10dd42f9821e4ba3048c` (36626d30).
Complete before/after configuration equality and root-enrolled original artifact
identities pass. Backup/install receipt: `/var/lib/kuno-prompt-processing-20261019`.
The transaction preserves ports, managed/image settings, model name and KNT2/KNR2.
No API/consumer, prompt persistence, distributed package, version or release change.
`native-prefill-installed-public-check-20261019/receipt.json` passes two actual
installed fixed-public code replies, each362 output tokens and identical text.
Cold load102.475s; first prefix6.296s/decode26.133s; warm prefix4.639s/decode10.899s
(33.21 output tokens/s, full warm request18.758s). All reply text remains in
protected client RAM. This is backend API acceptance, not fresh Vashti browser QA.

`native-prefill-installed-final-check-20261019/{receipt,host-final}.json` confirms
nonroot worker, actual installed model mount, swap/core0, zero worker write_bytes,
unprivileged inspection denied, zero OOM/driver faults, all three HTTP probes200
and normal idle admission. Native weights intentionally remain warm on Intel
(33,314,902,016B total device usage); NVIDIA compute0. This is not an empty-GPU
receipt: full physical recovery was separately proved before installation.
No staged service/test client remains active at the checkpoint. The bounded
sudo refresh loop is stopped after acceptance. No private user content was opened
or persisted; no new prompt-cache behavior is included.

## Future proposal: Vashti-owned portable prefix snapshots

The user asked about exporting processed prefix state to Vashti and supplying it
with later inference requests as an optional extension. This is a proposal only;
there is no API, storage or cache-retention change in this optimization pass.
For a fixed system/tool prefix, export would be needed once; subsequent requests
could upload that immutable snapshot. Growing conversation snapshots would need
an update/delta policy. Exact token-prefix/model/template/math-format matching,
import validation, bounded tensor decoding, encrypted Vashti storage, erasure and
capability negotiation need a shared contract before implementation. GPU handles,
raw pointers and unchecked caller-provided ownership receipts are not portable.

Approximate COMPACT selected-engine state, excluding weights, scratch, unused
32K allocation tails, serialization overhead and minor auxiliary state:
36 x48 x128 x128 x4 bytes =108MiB fixed GDN recurrent state, plus4.219MiB causal
convolution history. Twelve QSA layers contribute264x4 bytes/token each, pooled
128-float blocks per four tokens, and positions. MTP K+V add2x132x4 bytes/token.
Together this is approximately112.2MiB fixed +15,312 bytes/token:127.2MiB at1,024,
172.0MiB at4,096 and231.8MiB at8,192. PLE history, cursor/carry and format metadata
still need a complete serialization audit; these are estimates, not measured
exports. Gigabit Ethernet's theoretical125MB/s implies about1.5s for a180MB
one-way payload before overhead/restoration, potentially useful versus long
prefix computation. These are raw binary sizes: JSON/base64 alone adds about
one third. Lossless compression is possible but its ratio must be measured;
no reduction or generation equivalence is assumed. Cache sizes vary substantially
across architectures.

The processed state is sensitive derived user content, not model weights. It
must receive the same encrypted/private-lifetime handling as the source chat on
Vashti, and stay in protected RAM while imported by Kuno. System text is often
stable, but tool definitions/order, model template, dynamic date, exact token
boundary and conversation prefix may change it; a per-model name alone is not
a sufficient cache key. An absent optional snapshot must preserve current full
prefill behavior. An invalid/incompatible snapshot needs an explicit negotiated
error/fallback rule, not unchecked tensor use or silent wrong-prefix continuation.
A binary attachment/channel would avoid the base64 overhead; this is not yet a
chosen wire contract or a request to implement a Vashti consumer.
