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


## Accepted complete result

Source8a2652ecb4d76a5363728e8ad9b90e8114b4afa4, worker
e4d38a763a0498949107bce57d1dbaf452b58d64df06a838193f3e23369387fe.
Both preceding/local feature builds pass all390 AppCPU names, retaining387.
Actual compiler/CFG/negative/production AtomicOr classifier proofs pass in
both modes. Completed exact CPU selection/logs were reused for the SAME source
in the worker build; no duplicate AppCPU execution is claimed.18 feature-off
and59 full runtime,8+9 metadata and10 original native controls also pass.
All prior compiler populations and original native HAL source remain intact.

All983 GPU controls pass (original959 plus24 new sentinel controls). Both
original18-script normal/diagnostic full-model chains and all original
capacity/owner/status/finite/reuse/rolling/erasure checks pass. Genuine new
prefix counters remain2344 GU/2344 DOWN,287490 actual N128/WG256 workgroups;
all950 large-request dense products use wide32 with their finite consumers.
Whole accepted artifact includes actual source/executable/manifest hashes.

Diagnostic warm/first/repeated3.723993112/11.403487658/9.420896605s; normal
3.752159479/11.411931752/9.360428477s. Essentially unchanged from preceding
normal11.453113618/9.369922453s. No CLEAN rerun: normal remains slower than
retained8.854793s trigger. The best prior CLEAN8.394039699s and Strata
5.5006/5.3515s comparison remain unchanged. Goal ACTIVE/UNMET. Keep both complete
variants; do not abandon the coupled route/mapping change from these timings.

The intended scheduling change DID execute: large-request proven AtomicOr
views83798->88486 and actual encoded compute barriers108225->98849 (9376 fewer).
Legacy inventory230776 and copy barriers96 stay identical. Thus fewer barriers
alone are not the remaining whole-model gap. New dense/expert programs remain
0:0 spills:fills at128GRF;191 total ISA programs, original8 wide and2 QSA hashes
retained. Dense340/343 instructions90 sends; GU3043/553, DOWN2069/381.

Sampled layer24/start0/count4096 diagnostic matrix families (calls,ns):
cohort278/13354772; macro236/12887171; wideGU49/8075987;
wideDOWN49/3306294; tailGU48/3961142; tailDOWN48/1384476.
Dequant19238978ns and OrderedCopy29164373ns. These intervals can overlap;
do not sum them as elapsed wall time. Medium/small kernels retain their older
input-A/shared-stage arithmetic and remain a substantial matrix cost.

Protected unit `kuno-local32-complete-20261024`, invocation
5f89462c7b554cff9560372bb5dda059, exits0. CPU18m19.546s,32.2GiB peak,swap0.
Separate off/on CPU unit invocation16ab68f2f3af41fb9120e4b387457598 passes.
Independent04:06:13UTC state: Intel50,688,000B, NVIDIA compute0/display575MiB,
all5 classified kernel faults0; Kuno/Ollama/Vashti HTTP200, admission normal,
completed175. All benchmark processes terminal; physical GPU recovery passes.
No installation, service/settings/API/UI/driver/reset/reboot change.

Artifacts under `/media/jmicko/faststore/kuno-development`:
`local32-complete-driver-20261024.log`, `local32-unit-terminal-20261024.log`,
`local32-cpu-safe-20261024.json`, frozen `global-local32-root-*`, full983 summary,
`global-local32-{profile,normal}-safe-phases-20261024.json`,
`global-local32-complete-isa-export-20261024.json`,
`global-local32-final-state-20261024.json`. The earlier unprivileged build
rejection and3 independent mathematical-test artifacts remain preserved.
Bounded four-hour sudo refresher PID2712660 is still active; do not extend it
indefinitely. No credentials or private request content were persisted.

Next complete proposal: [global small/medium products](kuno-prefill-global-small-products.md).
