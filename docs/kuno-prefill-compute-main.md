# Complete compute-only primary queue candidate

Backend-only optional `strata-compute-main` / `--compute-main`. No API,
KNT2/KNR2, model, configuration, frontend or installation change. Existing
render-primary path remains separately selectable. This is a hypothesis about
queue/driver cost, not proof that all measured idle time comes from RCS.

The actual B70 exposes one queue in each of graphics+compute family0 (RCS),
compute-only family1 (CCS), and transfer-only family2 (BCS). Select the primary
by real nonzero compute-only capabilities, requiring two other distinct real
compute/copy families. Never infer queue1 or numeric family identities. Request
queue0 exactly once for all three families. The selected B70 roles become
main1/draft0/copy2; logical MAIN0/DRAFT1 scheduler lanes remain unchanged.
Missing/zero-count/aliased topology declines this candidate, without silently
falling back to the render queue.

Vulkan requires pipeline barrier stages legal for the actual command-pool
family (vkCmdPipelineBarrier VUIDs06461/06462). Cache immutable actual family
flags at HAL device adoption. Buffer/texture usage mappings keep every original
memory access and host/transfer/indirect stage; shader stages reflect actual
compute/graphics capability. Graphics-only index/vertex/attachment/present
uses are rejected on compute-only rather than deleting their dependencies.
Original graphics+compute mappings remain bit-identical. Actual inference
uses buffers; no new texture, render, ray query or AS functionality is claimed.
The uncalled AS mapper and unrelated swapchain/generic device setup remain.

Original device/queue adoption, native pools, timestamp validity, upload pools,
concurrent sharing across actual [main,copy,draft] families, host locks, owner
identity, timeline dependencies, original physical memory hazards, replay
signature and recovery all follow adopted indices. No new buffer, arithmetic,
retained capacity, allocation authority or request is introduced. Selected-family timestamp features additionally require its actual timestamp
width >=36 bits, rather than family0 advertisement. Root owns all
live tests; never reset/reboot GPUs.

CPU controls extend original171 with four actual topology/capability controls and four
actual production HAL mapping controls included directly from its shared module.
The latter cover legal compute shader stages, unchanged accesses, graphics
mapping parity, mixed host/transfer/dispatch-indirect, texture storage/init and
rejected graphics-only/transfer-only shader uses. Same complete241 GPU controls,
BOTH original profile/compiler/count/source/owner/erasure/recovery chains,
terminal/clean selection/staged quality/privacy/API gates follow. CPU/build/
GPU/speed acceptance is pending. Earlier Cargo dependency-package test invocations
were refused by workspace/dev-dependency selection before compilation; retain
logs and run the actual module from the application CPU harness instead.

A further positively identified first-use duplication remains for next work:
ordinary Wgpu pipeline validation+driver construction occurs before a second
native pipeline driver construction. Native dispatch has no ordinary error
fallback. Removing unused construction must preserve the original Wgpu device,
entrypoint, interface/layout and device-limit checks, not substitute a weaker
host reflection guard. No such optimization is implemented in this checkpoint.


## Frozen validation checkpoint

Source69c045a73e307090e84f1f9ba81b362a3095435c, artifact
`strata-parity-compute-main-20261022`, SHA256
306d4618638cd626615ee2335969fa21c138d11bea4196baa5a5618f9217e6fe.
Exact-source179 application CPU controls PASS (original171 plus four topology/
timestamp and four actual production HAL mapper controls). Separate original11
runtime grammar/owner controls PASS; runtime tree unchanged fromaf4de6f0.
All421 WGSL and248 matrix layouts PASS original pinned2026.1 validation.
No numerical/kernel/operator/layout inventory changed.

Root protected unit`kuno-compute-main-20261022.service` now runs complete
original241 GPU controls and BOTH original proof/profile chains. Wrapper
`check-native-compute-main-connected-20261022.py` SHA256
12b712274d8a68a6be78ee1f27d47d7be24e1eac0344850224d3ecade5308288
pins exact source/artifact/feature list/all179 original CPU names and11 runtime
controls before hardware. Original reference/count/compiler/source/owner/
retained-capacity/erasure/recovery proofs remain. Evidence in the usual base:
`native-compute-main-{cpu-exact-final,runtime-cpu,offline,build-all,connected}-20261022.log`
and`native-compute-main-runtime-unchanged-20261022.diff`.
GPU correctness/performance/terminal/clean/staged/install acceptance pending;
installed worker/gateway unchanged, no Strata parity claim.


## Complete connected outcome

All original241 GPU controls and BOTH complete original73/4245/4245 profile/
compiler/FMA/count/source/owner/retained-capacity/erasure/recovery chains PASS.
Exact full proof `strata-parity-compute-main-20261022-compute-main-public-summary.json`;
`native-compute-main-connected-20261022.log`. Original source/math/layouts and
logical scheduler lanes remain. The protected unit completed/inactive/dead.

Normal first/repeat11.970535604/10.533478310s versus RCS global-narrow
10.653114762/9.164950382s:12.37%/14.93% slower. Sampled first/repeat
11.967792619/10.585019376s also slower. Normal warm expert5.961181356s and
attention3.300396782s versus4.942176667/2.953429977s; capture is0.605961328s
HOST WALL, not CPU utilization. This is a complete correct optional path,
not selected for installation. Retain it and use render-primary for the next
independent native-prepared candidate. No terminal/clean/staged/install gate
was run for this slower candidate; no speed/parity/decode gain is claimed.

An additional normal-schedule fixed-public cold flat CPU+PMU diagnostic is
running in`kuno-compute-main-first-costs-20261022.service`, using the complete
source-pinned proof above. Helper SHA256
a7818cd44d6485e78eb8726649d10f169ac9b35ae6104b8481755da181846944
retains all flat symbols (no percent cutoff) and requires actual CCS activity.
Raw instruction pointers and counters stay in root-only/run tmpfs and are
removed; no stack/register/payload samples. It cannot serve as a clean speed
result. No GPU timestamp commands, schedule serialization or parallel builds.
Installed worker/gateway remain unchanged. No GPU reset/reboot is permitted.


## Actual compute-engine diagnostic completed

`strata-parity-compute-main-first-costs-20261022-cpu-proof.json` confirms original
three clean public replies/CLOSED, controlled interval, zero lost CPU samples,
full symbol report, no added GPU timestamp or schedule serialization, raw tmpfs
erasure and physical recovery. Unit completed/inactive/dead. First-long interval
13.2836s: RCS0%, BCS14.9273%, CCS56.0231%. The actual engine changed as intended;
at least29.05% had no counted compute/copy engine active. No gain from queue
selection, and no clean-selection benchmark claim from instrumentation.

Full flat report `strata-parity-compute-main-first-costs-20261022-cpu-symbol-report.txt`
retains1601 symbol rows versus the earlier percent-cut report. Approximate rounded
DSO shares sum101.75%, so do not present their sums as exact percentages. Roughly
56% of displayed weight is anonymous Vulkan code,28% application/IR/runtime,16%
libc. No anonymous driver offset is labeled as JIT without symbols. Positively
identified work includes original IR cloning and planner/binding; duplicate
ordinary/native driver executable construction is proven separately in source.
Earlier first helper failure and all optional paths/receipts remain intact.


## Exact driver symbol attribution

The anonymous driver offsets are now resolved from the exact published Kisak
debug package, without system installation. Evidence directory:
`/media/jmicko/faststore/kuno-development/intel-driver-symbols-20261022/`.
Installed library SHA256cbbcd568c7b39fa2789463c8e2ef6414b023b1f86435f48baef705aabcfae1f2,
ELF buildID9e30c0f3d8ad967dbeaecdb782a2f3f030cb05a1; debuglink CRC8cf845d5.
All three identities match extracted debug SHA256
99bfe5a6ad02b0adbb31a8ccf38d56b0bc6f38678f6de1be14daf69f45eee7f2.
Package SHA2562c336ba4f97d089c75c5626e39d256bd6d3e72a0be68d61c17f026fb77fd2bb7
was checked against the primary Kisak main/debug Packages.xz entry before use.
See`provenance.json` for URLs, exact sizes and supplementary DWZ identities.

Offsets8d0eb4/8d0a80/8d0ec9/8d12e7 are Intel compiler
`propagate_block_scoreboards`;744c0 is`anv_cmd_state_init`,7498d its command-buffer
reset caller. `flat-symbol-table-aggregate.json` independently resolves all887
driver offsets through bounded exact ELF symbol ranges: already-rounded total
56.50% driver,11.23% scoreboard propagation,5.29% live-variable computation,
1.43% live-variable start/end. These are positively identified shader-compiler
functions; do not call every other driver symbol JIT or these exact CPU seconds.
GNU addr2line source-line lookup warned about an absolute supplementary DWZ
path; matching supplementary debug was also extracted, never installed. Hot
function identities are independently confirmed without any DWARF dependency.
This supports the separate native-prepared duplicate-executable target, without
claiming how much wall time that change will save before hardware measurement.
