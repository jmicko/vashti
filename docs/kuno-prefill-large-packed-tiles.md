# Packed large-expert operand tiles

Proposed coherent producer/consumer candidate based on diagnosticf1cc4445
(math-identical to accepted8ab19c0e). No GPU acceptance or speed claim yet.
Read the direct-profile report and current coordination handoff first.

## Evidence and intended change

Same public4096-token/layer24 interval: Kuno large matrix12.069422ms versus
Strata3.794023ms; different profilers and slightly different router populations.
Current source's large prefix invokes local_products with strided global operands.
Pinned Strata actual oneMKL native kernels use block2D loads, wider register tiles
and256GRFs; the current Mesa path uses128GRFs and scattered row-stride loads.
The same dpas.8x8 arithmetic instruction is present in both. These observations
identify a lead, not a guaranteed speed gain or a Rust-language limitation.

Pack both operand producers with their consumers. Preserve all original math,
K16 chronology, route order, source tiers, finite/status/fault and ownership rules.
No oneMKL/native blobs/SDK dependency, driver modification or new Vulkan extension.
The original row-major path remains independently selectable and testable.

## Exact layout and owner contracts

Use tiles of16 reduction columns. An input tile has16 member rows; a weight
tile has8 output rows. For row r, reduction column k, reduction extent K and
row-block B=16(input) or8(weight), half offset is
`(r/B)*(B*K) + (k/16)*(B*16) + (r%B)*16 + k%16`.
The map is a bijection over padded rows divisible byB and K divisible by16.
All8-half dequant jobs and2-half image/activation jobs retain contiguous outputs
and cannot cross a16-column tile. Packed and original logical byte sizes agree.
Original parent offsets are applied OUTSIDE the per-group permutation.

Only large groups (members>=256) in the global layer X owner use the new layout.
Small/regular group segments remain byte-identical row-major. The layer producer
has a distinct name and its own authenticated format1 parameter. Global group
prefixes and Pad32 zero gutters are unchanged. Every full/partial large group
segment, including its gutter, must be completely initialized and tested.

Large-only GU/DOWN FP16 weight images use8x16 tiles. Large activation writes H
in16x16 tiles after reading the unchanged F32 GU layout. Large prefix and ALL
8/16/32 residual consumers address these packed inputs/weights. Their existing
shared staging for tails unpacks through the exact index, and their original
publication/reduction stays unchanged. The prefix uses stride16 cooperative
loads and the original eight chronological C chains. No isolated format toggle.

INFO word14 is the explicit packed-large format (0 original,1 packed only with
new feature AND layer mode1); word15 remains reserved. New producer and matrix
identities require1; old identities require0. The layer grammar retains a format
seal and rejects a packed unit following an original layer producer or an original
large unit following a packed producer. Small/regular units can consume only their
unchanged segments. All actual owners, source dependencies, phase/raw releases,
unit/layer closure and real grids remain enforced. No alias of a historical name
receives new role credit. No hidden memory/view widening.

Owners remain global X277.5MiB, GU20MiB, H5MiB, images75MiB/two32MiB raw banks;
MAIN/root charges, original512MiB driver margin and11983resident pairs unchanged.
No input/output/model content is persisted or added to private telemetry.

## Coherent scope and acceptance

Feature strata-large-packed-tiles/--large-packed-tiles requires the entire
layer-gather/large/dual/regular bundle and corresponding runtime/wgpu feature.
Compose global X, direct8 images, H activation, every large matrix/tail consumer,
strict native identities/format/owner grammar, real profile counters and build/
compiler/oracle integration BEFORE any full-model speed decision.

Retain all original2085 GPU and456 AppCPU controls and all runtime/selector/
compiler inventories. Add independent exhaustive layout bijection/inverse,
vector-boundary/group-prefix/zero-gutter checks; generated WGSL and actual matrix
SPIR-V geometry/owner/guards; paired full-owner GPU producer/dequant/H checks;
all large prefix/tails finite/poisoned/nonzero-prefix cases against original
arithmetic and unchanged F64 bounds. No relaxed old oracle or validation limits.
Then both full original model chains, source/copy/erasure, original CLEAN
45/4245/4245 comparison and independent physical recovery. No numerical, speed,
installed/API/Vashti/release claim at source completion. Hold medium C4 candidate.
