# K32 expert workgroup lifetime reuse

Optional backend source based54162b57. This component is unbuilt and untested;
root owns the complete build, actual compiler/GPU proof, whole-model acceptance
and any later performance comparison. No service, GPU, installation, model,
public API, storage, context or Vashti change was performed by this subtask.

## Coupled mechanism

Keep every original48KiB producer, definition and public oracle available. New
genuine entries share the original K32 operand stage, next-register prefetch,
chronological two K16 product groups per K32 stage and publication helpers.
The immutable header predicate has a genuine plane_all vote, as in the guarded
deep kernel. The old scalar while loop is expressed as a scalar uniform
for stage_index in0..columns/32. The final original whole-workgroup handoff
ends every operand reader. Only then are BOTH A/B SharedMemory objects freed,
and only afterward is the original16KiB publication scratch allocated.

The actual CubeCL SharedLiveness/Free mechanism assigns non-overlapping live
slices. Its real explicit-workgroup-layout SPIRV uses Aliased Blocks and actual
byte offsets: A/B are two disjoint16KiB arrays; later F32 scratch reuses a dead
operand range. Physical peak is32768B in the explicit compiler and49152B in
the implicit compiler. The HAL canonicalizer preserves those actual Block
offsets; it does not invent aliasing. Runtime selection requires the actual
client's explicit-layout capability AND hardware>=32768B, original complete
shape, alignment,512-thread/SG16 and CMMA capability, all original bound owners,
prefix/tail preflight and grids. No64KiB deep-matrix dependency is needed.
Unsupported capability falls back through the unchanged original dispatcher.

All original GU checked/direct, rank DOWN and ordered pending2 publication
helpers are reused. Invalid-route fragments retain their exact checked finite
fallback. Conditional subgroup fallback requires the genuine existing subgroup
publication fences; the feature must imply strata-native-subgroup-publish and
strata-expert-ordered-output. The final whole-workgroup operand handoff is
separate from those later subgroup publication fences.

## Actual entries and production selection

Genuine types, all retaining original six owners and seven runtime u32 scalars
(columns, rows, count, weight_first, group, member_base, piece_members):

- native_expert_wide::shared_reuse::expert_wide_gu::ExpertWideGu
- native_expert_wide::shared_reuse::expert_wide_gu_direct::ExpertWideGuDirect
- native_expert_wide::shared_reuse::expert_wide_down::ExpertWideDown (rank oracle)
- native_expert_wide::ordered::shared_reuse::expert_wide_down::ExpertWideDown

Each uses512 threads, SG16, original128x128 prefix geometry. The existing old
GU/rank DOWN tails and original ordered DOWN tail remain unchanged. A new
prefix with its matching ORIGINAL tail is a legitimate mixed identity; rank
and ordered DOWN tails may never be interchanged.

Top-level production expert/expert_ordered select only while the genuine native
expert capture is open. Original run_expert, try_expert_mode and old public
entrypoints retain their old selection. Successful actual new prefix dispatches
increment separate public-only expert_shared_reuse_gu_prefixes or
expert_shared_reuse_down_prefixes. Eligibility alone never increments them.
The original exact counter laws and original source populations remain.

## Additional proof populations

Four additive CPU controls cover actual32KiB/explicit geometry and all original
shape limits, token-identical original prefetch/CMMA body and full chronological
stage order, last handoff/free/late scratch and original fallback preservation,
and all genuine distinct kernel identities with untouched old definitions.

validate_shared_reuse_sources separately compiles four genuine entries across
all eight actual compiler-layout combinations,32 modules total. It checks the
actual emitted shared_size, typed original16KiB A/B and F32 arrays, real Aliased
Block offsets and actual scratch overlap with disjoint operands, original
subgroup16, eight CMMA loads/eight products, publication and sticky faults,
existing register-stage CFG, canonicalization and spirv-val. Its distinct marker
is strata_wide_shared_reuse_shader_accepted. No original validator is changed.

public_shared_reuse_numeric adds48 raw producer controls for the current bundle
WITHOUT native-expert-medium: six controls x four families x original/new.
Controls: count256 finite, count383 finite (real127-row tail), count257 column0
NaN, count257 other-column NaN, count257 invalid prefix route, count257 invalid
tail route. Families: checked GU, direct GU, rank DOWN, ordered DOWN. A separate
medium-enabled build adds16 controls at counts128/129; do not report these for
a build without that feature. The existing original full-owner/F64 fixture is
reused with the same numerical threshold, original grouped map, offset weights,
all output/status/fault/source and padding owners. Direct raw GU and ordered
pending2 are checked under their actual raw-before-consumer semantics. Existing
old cases cannot select the new raw-GU mode. The original consumer controls and
both original full-model chains remain required for complete pipeline proof.

No speed conclusion follows from source completion or a single compiler/module
sample. Compare only after the entire coupled candidate and original controls
pass, then confirm actual production counters, ISA and physical recovery.

## Complete composed validation, 2026-10-23

The complete `c501ab88`/`f9a3725b2eba522e209c9737228c632fc0178e37da3c77c19a8a9f66837af18b`
bundle passes326 selected AppCPU,18/34 runtime controls,8 original metadata
controls, all original shader populations plus32 shared-reuse/30 direct8
modules,96 register CFG checks,573 GPU controls and both original18-script
whole-model chains. Real selection counters and physical recovery pass.
Diagnostic normal repeated prompt8.745485045s improves preceding8.992040901s
but does not match Strata5.3515s. No clean selection or installation. See the
[active coordination handoff](kuno-vashti-coordination.md) for exact evidence,
limitations and preserved initial build/fixture failures. Earlier source-only
status in this document describes the isolated implementation stage.
