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
