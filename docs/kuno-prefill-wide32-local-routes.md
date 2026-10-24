# Local route selection for the complete wide32 bundle

Backend-only laboratory candidate, composed on d081b38a. No installation,
Vashti/API, capacity, precision, ownership, status or erasure change.
Feature `strata-wide32-local-routes` / build option `--wide32-local-routes`
requires the complete `strata-global-wide32` bundle. The preceding complete
variant remains selectable without this new feature.

## Confirmed source difference and coupled change

The current wide32 expert shader uses an atomic load of the shared sticky
fault to select raw versus checked arithmetic. The existing backend's actual
SPIR-V `atomic_or::mask` proof deliberately refuses descriptors with ANY read.
Thus this shader cannot use the existing commuting, unused-result AtomicOr
dependency treatment for its fault owner. This is a source-level fact; its
whole-model performance impact has not been measured. Strata's active oneMKL
GEMM does not read that shared fault. Do not relax the backend proof.

Instead, each subgroup authenticates its own two sets of16 real routes, then
uses subgroup All to select raw eight-C arithmetic or the original checked
publication. Keep every header and real count/prefix/role check, all original
Gather/finite/activation/inverse consumers and the sticky AtomicOr updates.
There is no fault read, clear, extra flag, extra production dispatch or host
attestation of GPU contents. Earlier unrelated faults still remain sticky but
do not change this producer's valid arithmetic, as in the original producer.
Standalone numerical guard dispatches remain unchanged for comparability;
the new arithmetic no longer relies on their result to select its branch.

CRITICAL coupled mapping: checked serial subtiles MUST cover the SAME32x32
cell as that subgroup's raw branch. Use token=Y*128+(sg/4)*32+tile_m*16 and
row=X*128+(sg%4)*32+tile_n*16 for two-by-two serial subtiles. Keeping the old
whole-workgroup64x64 subdivision while making branch selection local would
overlap or omit output cells. Both mappings change together.

Keep transposed two-C fallback and its exact8-member failure/finite granularity,
nonzero invalid-route sentinels, pending2 (not accepted output), original8KiB
compiled scratch, chronological K16 products, real N128/WG256 grid and tail.
All6 owners and7 scalars, native closed grammar and tape limits stay identical.

## Acceptance

Retain all387 CPU,18/59 runtime,8+9 metadata,10 native and959 GPU controls.
The three added CPU controls make390 in each feature-off/on build. Six added
nonfinite/sentinel controls at members16,24,32,63,127,255, with both roles and
both old/new producers, add24 GPU cases for983. Reports explicitly carry the
invalid member so repeated control labels cannot conceal missing positions.
Test source with the feature both absent and present. Add independent exhaustive
mixed-subgroup coverage (all65536 branch masks), original-half route/sentinel
semantics, actual compiled branch CFG and map operand tracing, and invoke the
EXISTING production AtomicOr classifier on actual emitted modules. It must
classify precisely original fault binding5 for the new variant, and refuse it
for the preceding atomic-load variant. Mutations to subgroup All, route-map
binding or introduction of a fault read must decline the new proof. No guessed
shader names may grant access/dependency authority.

Only complete numerical and full-model/recovery evidence can accept the
candidate. Speed interpretation remains the whole first/repeated comparison;
the unchanged clean Strata target is5.5006/5.3515s. No performance claim yet.
