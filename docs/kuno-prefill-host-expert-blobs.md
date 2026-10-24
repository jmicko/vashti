# Contiguous HOST expert blobs

Next source plan after complete granular acceptance source2391584c. NOT yet
implemented, compiled or measured. Keep the full accepted granular/interleaved
candidate and every previous variant. Target remains Strata clean first/repeat
5.5006/5.3515s, same model/context/public fixture. No API, Vashti, install,
release, driver or reset change.

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
