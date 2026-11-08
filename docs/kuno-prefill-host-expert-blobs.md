# Contiguous HOST expert blobs

Complete validated source candidate after granular acceptance source2391584c.
Frozen sourcee2279c48 / worker1e77aa78 passes fresh source, numerical and full
model/recovery checks below. No measured prompt-processing gain. Keep the full accepted granular/interleaved
candidate and every previous variant. Target remains Strata clean first/repeat
5.5006/5.3515s, same model/context/public fixture. No API, Vashti, install,
release, driver or reset change.

## Completed acceptance, 2026-10-24 UTC

Runtime source `e2279c48cdd87cd66d8942d0b484e151cd09975a`, worker SHA256
`1e77aa78662be507f9b079e00243d688f5ecc3b3bbcf78f935ba10fdcc1ed4a5`.
Full variant/legacy constructors remain selectable. Nothing installed or released.

- All435 selected AppCPU controls pass: previous425 plus ten new controls.
  Same-source feature-off425 also pass. Original18/91 runtime,8+9 selector,
  ten native metadata and30 new synthetic counter controls pass. Actual old
  shader/resource controls and all48 granular layouts pass.
- Fresh ALL1987 GPU math controls pass, with BOTH complete original18-script
  model/compiler/counter/owner/erasure/recovery chains. No numerical receipt
  reuse. The unused reuse prototype was rejected before execution: exported
  SPIRV sets were not byte-identical, and equivalence was not established.
  Keep the old receipts and failed/prototype artifacts; do not relabel them.
- Fresh compiler capture has240 programs, six genuine granular programs with
  zero spills/fills. Both QSA reference/optimized ISA hashes remain exact.
  Do not claim all old/small compiler chunks are byte-identical.
- Real cold public Session audit compares EVERY25878092800 HOST expert byte
  with original GGUF, covering12593 experts through350 actual scoped CPU maps.
  Pinned-model logical padding is zero. All435 real registered-HOST to original
  raw-bank boundary cases pass, including907084800 payload bytes and2610
  rejected view/stream/owner/role-size/cell/extent cases. Both whole raw roots
  are restored and all67108864 bytes zero-checked;1403217408 transient I/O bytes
  are zero-audited. Checked Session.close, CLOSED frame and physical recovery
  pass. No extra GPU roots. Physical adaptive rebase was NOT performed.

Actual source quotes remain11983 resident experts/24414233600 GPU bytes and
25878092800 HOST expert bytes;26215808000 HOST bytes including embedding.
Registered HOST roots288->234; all original limits/grants remain. Repeated
fixture still has13291 streamed groups and27345664000 transferred bytes.
Submitted HOST copy ranges39873->13291, with26582 saved ranges and13291 genuine
whole-blob groups. These are submitted ranges, not claimed hardware DMA counts.

| Scope | Warm prompt | First4245 | Repeated4245 | Initial load |
| --- | ---: | ---: | ---: | ---: |
| Normal host schedule | 3.791275980s/73 tokens | 10.880471877s | 8.852502616s | 78.715295570s |
| Sampled | 3.812114518s/73 tokens | 10.963030043s | 8.998811837s | 80.304535593s |
| Original CLEAN | 6.552246114s/45 tokens | 12.766184520s | 8.427964917s | 76.999806388s |

All requests produce four checked output tokens; no sustained decode claim.
These are individual observations, without variance control or disk-cold load
control. Normal repeat is0.18% shorter than previous8.868221228s; sampled repeat
is longer. CLEAN12.766184520/8.427964917s is longer than preceding complete
branch CLEAN11.817666824/8.323282816s. No speed win or parity. Strata CLEAN
5.5006/5.3515s remains unmet. The original normal trigger8.854793s was retained;
8.852502616s triggered ONE original CLEAN, not a favorable retry.

New normal repeated expert/attention intervals5.225996740/2.420706493s. Sample
layer24/main4096: copy28.419784ms, dequant17.730710ms, macro19.271555ms,
cohort7.049892ms, wideGU8.025676ms/wideDOWN3.354319ms. Sample intervals cannot
be summed/scaled into normal wall time. Next audit
[regular expert cohorts](kuno-prefill-regular-expert-cohorts.md), composing the
whole producer/consumer/native-grammar/counter pipeline before timing.

Artifacts under `/media/jmicko/faststore/kuno-development`:
`host-blobs-root-build-receipt-20261024.json`,
`host-blobs-real-bytes-safe-20261024.json`,
`host-blobs-v1-{acceptance-freeze,complete-isa-export,normal-safe-phases,profile-safe-phases,original-clean-safe,final-state}-20261024.json`,
`host-blobs-independent-final-state-20261024.json`, and protected
`strata-parity-host-blobs-v1-20261024-{host-source,operators,profile,normal,clean}`.
Full supervisor `kuno-host-blobs-complete-v1-20261024` exits0. Final independent
state is recorded separately; all five fault classes zero, HTTP200 on all three
services, no recovery latch/queue/pause, Intel42278912B and NVIDIA compute0.

Preserved initial failures: source5770d07d did not forward the public diagnostic
selector into serve and had an untyped test bitmask; d1637f2a fixes forwarding,
3875ed8d completes bitmask typing. Its copy fixture then exceeded its own64KiB
registry metadata quote. e2279c48 correctly charges the original bounded1MiB
metadata cap in the test fixture; no production limit changed. Full AppCPUv4/
offv1 and worker/recovery receipts bind the final e2279c48 source.

## Confirmed difference and motivation

Pinned MIT Stratafb58e0dbc8399662c0e47c76578c6e878b14f6cf uses a pinned HOST
mirror with complete `[gate|up|down]` blobs. Active non-stream-all
`sycl/src/prefill/prefill.cpp:3830-3848` issues one memcpy of `lay.blob_bytes(l)`
per streamed expert. Read the earlier [source/placement comparison](kuno-prefill-source-placement-comparison.md)
and [original graph contract](kuno-native-text-graph.md#full-routed--shared-moe).
Do not describe Intel's default as fused CUDA quantized MoE: its selected path
is FP16 dequantization plus oneMKL GEMM.

Kuno's actual selected `PromptLayer::stage_group_on` calls
`Sources::stage_prompt_group_on`, resolving and copying all three separate HOST
role ranges. Current HOST pages are role-major; source registration and CPU
borrowing use `(layer,role,piece)`. This is not a pageable/file-read path.
The accepted repeated fixture has13291 streamed experts,39873 role copies and
27345664000 copied bytes, unchanged from the preceding source. Actual HOST
expert logical bytes25878092800; including embedding, allocated/registered
HOST bytes26215808000 across288 buffers. Resident11983 pairs/24414233600B,
46 GPU roots; do not reduce residency to fund duplicate HOST copies.

Current sample layer24/main4096 has ordered-copy29.178992ms, macro19.282232ms,
cohort7.064579ms, dequant17.714503ms, wideGU8.026197ms/wideDOWN3.327654ms.
These sampled serialized intervals must NOT be summed/scaled into normal wall
time or compared as identical scopes to Strata's residual `wait copy` phase.
They motivate source/copy work, not a guaranteed bandwidth or speed claim.

## Compose the whole source layout and consumers

Add an explicit complete variant, tentatively `strata-host-expert-blobs`, over
the accepted granular bundle. Preserve legacy constructors/role-major mode for
all existing controls and selectable variants. Do not select/rollback isolated
pieces based on timing.

Plan complete immutable HOST blobs per nonresident expert, in original missing
expert order. Each blob's role bytes/order must match the existing raw slot and
resident blob ABI exactly. Keep every expert wholly within one real registered
HOST root. Preserve original128MiB HOST piece limit,256-byte alignment, source
window/setup bounds and all global HOST/registry/admission caps. Chunking is
allowed; a single26GB Vulkan allocation is not required. Never increase an
advertised reservation without charging the actual backing, padding, metadata
and simultaneously live setup windows. No extra full model/HOST mirror.

Keep original ranked resident prefix, per-expert role byte geometry, graph/file
binding, virtual expert mapping, CPU/GPU placement and device source-table ABI.
Plan all genuine HOST offsets from the new layout. Reuse the current Source
representation if possible; a packed base page may use a distinct internal
canonical role tag, but must have ONE PageOwner/registry authority. Do not make
three PageOwners alias one allocation or narrow the whole-root lease.

Compose together before model timing:

- Actual plan/chunk/source calculation, complete byte/padding quotes and exact
  role range mapping, preserving graph binding and old constructors.
- Bounded parallel/public GGUF reads into FINAL allocated HOST spans; publish
  registry authority only after every reader joins. No on-request repack/read.
- HOST `role`, complete `handles`/read leases and source-table/prototype binding.
- CPU borrow/map/role lookup: deduplicate by the actual physical root, so one
  allocation is never mapped twice for its different logical roles. Preserve
  sole CPU borrow and all retirement/mutation exclusion and lifetime guards.
- Public CPU/GPU placement/rebase/exchange, including transfers/restoration,
  accounting and revocation. Audit every actual consumer; not just prefill.
- Optional prompt-loan shadows and embedding: do not reinterpret their separate
  owners as packed base pages or bypass current effective-source resolution.
- Existing raw-bank factory/stream/slot/offset/extent checks and full erasure.
  First resolve/validate all CURRENT role sources, including shadows. Coalesce
  only when all actual source bindings are the SAME registered whole root and
  exact adjacent source/destination spans. Otherwise retain original role copies.
  No coalescing across roots, gaps, types, stale authority or phase changes.
- Honest actual segment/byte counters and a new explicit helper branch. For
  selected no-loan all-HOST base blobs expect one copy per miss; preserve every
  byte/group and separate map/inverse copies. Do not credit three old copies.

No dequant/matrix precision, image layout, producer math, route/status predicates,
ordered DOWN, source releases, packet command limits or GPU-root budget change.
The raw ring, dequant ring and all completion/quarantine gates remain original.

## Acceptance and limits

Finish the WHOLE owner/layout/consumer/copy/counter variant before speed tests.
Retain all425 existing AppCPU and18/91 runtime controls and all1987 math controls.
Add actual-source CPU range/coverage/padding/boundary/graph/quote controls, real
role/owner alias negatives and CPU-map/rebase/shadow/retire cases. New test names
and populations must be frozen only after inspecting actual implementations.

New physical evidence must exercise the changed source producer and consumers:
verify complete loaded public model bytes and padding against original GGUF
through bounded read windows, and genuine registered-HOST -> enrolled raw-bank
one-copy results against all three original role ranges. Include root/chunk/
role boundaries, changed ownership/offset/stream, partial effective shadows,
completion/retirement and recovery. No private inference payload may enter logs
or disk; these tests use only original public model bytes/fixed public prompts.

Existing synthetic matrix controls do not validate this HOST layout. Since
kernel math is intentionally unchanged, avoid an unnecessary fresh replay of
1987 unchanged synthetic cases IF actual unchanged shader/resource/source
provenance is proved and the reuse is explicitly recorded as reuse. Rerun the
relevant numerical tests if compiler/kernel sources, types, values or image
consumers change. Never claim an old receipt came from a new worker.

Then both original complete model/counter/owner/erasure/recovery chains, genuine
new producer counts and the original CLEAN first/repeat gate. Same model,
context, prompt/seed and sampling remain fixed. Keep all old capacities and
thresholds; a sample or expected memcpy reduction is not speed acceptance.


## Initial composed source checkpoint

`strata-host-expert-blobs` / `--host-expert-blobs` requires the entire accepted
granular bundle. Legacy plan constructors remain role-major; the real graph
constructor selects packed HOST chunks only under this flag. Complete-expert
256B strides stay within original128MiB HOST roots. HOST registered spans and
backing/setup/metadata quotes include real gaps; original source bytes and GPU
resident prefix remain unchanged. Each packed page has ONE original registry
owner with a private canonical base-role tag. CPU map lookup deduplicates this
root; shadow/embedding lookup remains independent. Existing adaptive rebase
revokes all base pages by layer, and its inventory resolves through the same
actual HOST role method, retaining whole-root ownership and aligned role ranges.

Live original header SHA verified against the pinned digest: all144 routed role
tensors are in original shard0 (the second shard contains the PLE embedding).
The loader independently requires all three roles in each packed page to share
the same authenticated shard. No mixed-source provenance fiction/schema change.
Scoped readers fill FINAL registered spans, retaining original reader join and
full-backing initialization/cache flush/publication. No request-time repack.

The actual raw-bank factory resolves current sources first, then the new pure
coalescer validates all original bindings/ranges, merges only genuinely adjacent
same-root spans, and revalidates the result. Backend registry lookup, phases,
limits, owning stream, copy scheduling and completion remain unchanged. Actual
copy/byte and new saved-segment/blob-group counters reflect submitted ranges.

Dedicated `--public-host-expert-blobs-check SETTINGS` accepts no request text.
It loads the real normal cold Session, verifies EVERY loaded HOST role byte
against original GGUF through bounded windows and actual scoped CPU maps, then
checks real HOST->both original raw banks at all packed-root boundary experts.
It exercises rejected changed owner/view/stream/role-size/cell/range requests,
checks copied role bytes and neighboring zeros, restores/zero-audits both full
raw roots and sanitizes tracked transient I/O. Original checked Session.close
still owns detach/complete erasure, including on error. No extra GPU roots or
private entry diagnostics; physical rebase itself is NOT claimed by this probe.
Its live execution/Cold-owner cleanup still require validation before timing.

Planned initial CPU additions: three actual plan/range/alignment/quote controls,
three exact real-registry-metadata coalescing/negative controls, one actual CPU
map-key/shadow exclusion control, two composed reader/padding/failure-join
controls and one exhaustive explicit-public-entry guard. Expected435 full/
425 feature-off selected AppCPU; freeze after actual compilation/test listing.
Original18/91 runtime,8+9+10 metadata/native and every shader/resource control
remain required. No new vendor feature or matrix/dequant shader change.
