# Medium expert tile and native vertical swaps

Complete optional source candidate,2026-10-22. Not tested or installed yet.
No API/KNT2/KNR2/context/model/settings/frontend change; required Vashti:none.

Current same-host repeated4245 prompt diagnostic remains about9.42s; clean
Strata5.3515s remains the goal. `kuno-prefill-expert-gap.md` records source
comparison, sampler limits and old complete candidates. In the sampled public
layer24/start0/count4096, original32-row macros take236calls/13.053ms and
small cohorts278calls/10.826ms. These are not full-prompt critical-path times.
Strata uses oneMKL; this is a sizing hypothesis for our Rust/CMMA backend, not
an assertion that oneMKL uses this exact kernel or a proven gain.

## Complete coupled selection

`strata-expert-macro64` / `--expert-macro64` retains all prior expert/cohort/
raw-bank/register-prefetch and QSA prefetch/scale/native-XOR implementations.
It explicitly requires `strata-subgroup-quad-vertical` /
`--subgroup-quad-vertical`, itself requiring complete nativeQuad and predecessor
features. The complete horizontal+vertical implementation remains selectable.

Full nativeQuad b7262ce6/worker9b4f090f passes232CPU/runtime13/offline429/280,
all330GPU operators, and the sampled whole-model computations with recovery.
Its native code has212 indirect moves versus332, but35spills/81fills versus0/0.
The original exact compiler guard correctly rejects it; normal/both-chain and
clean/staged/install acceptance were not completed. Sampled first10.272306824/
repeat9.222505885s and attention2.738178453s are diagnostic only. Preserve the
failed full-chain log and original zero-spill guard. This is no accepted speed
result or partial success relabeled complete.

Vertical mode replaces only26 genuine constant XOR2 scalarF32 operations with
QuadSwap vertical, preserving26 XOR1 and78 XOR4/8/16 operations plus48 indexed
PV gathers. Horizontal swaps need two strided temporaries in Mesa; the vertical
mode targets that measured register-pressure constraint. All original arithmetic,
participant order, source/owner/fault/precise-product and hardware-property gates
remain. Exact feature-specific opcode controls require26Quad/104XOR/48gather,
versus52/78/48 for the retained full mode and0/130/48 for XOR-only mode. One new
CPU control independently inventories all masks; no spill or numerical bound
is relaxed. Actual assembly must still show0/0 spills/fills.

## Real64-row expert products

Only authentic groups64..255 with original fixed GU(1280x2560) or
DOWN(2560x640) shapes, real aligned vector bindings and at least49152B hardware
shared memory select the new kernel. Other groups keep original paths. The
original32-row macro functions and source are unchanged. New64x64 products use
128 threads/eight16-lane subgroups, four original8x16 accumulators per subgroup
and orderedK16 steps insideK64 stages. Staging copies all actual input64x64 and
weight64x64 values as original vec4 FP16 values. Masked input tails are zero;
no input after the authentic real-member extent is read.

Double operand buffers are32768B and output publication4096F32/16384B.
Actual logical envelope49152B; explicit-lifetime compiler layout may reuse
fenced storage for32768B. Both prices are separately checked; no advertised
capacity reduction or missing owner is hidden. Every first-stage, handoff and
publication barrier remains. Original GU finite/zero/fault and route-scattered
DOWN/status publication are unchanged. Map/header/prefix guards run uniformly
before cooperative work. Canonical input/weight/output owners and six logical
phases remain; this is not a new cohort or a wide-prefix fallback.

The native grammar admits the distinct generated Macro64 type only with exact
64-row grid and64..255 members, retaining complete INFO/role/source/owner/order
checks. New actual runtime controls cover all six phases and reject original
32-row grids, altered owners, counts outside selection and bad grids. A fixed
public counter records actual queued products; every long production request
must show exactly2*(groups64_127+groups128_255), not a guessed count.

## Required evidence

Retain232 prior App CPU controls plus genuine vertical mask inventory and four
macro64 controls: hardware/shape/binding/envelope selection, unique stage and
masked-tail source addresses, complete output lattice and chronologicalK16,
and real two-role/eight-layout compilation with128threads/16original products/
three barriers plus actual generated metadata. Intended237App controls.
Actual runtime15 retains all13 and adds two real grammar controls. Full offline
429 WGSL and296 matrix layouts retain every old429/280 and add16 genuine layouts.

All330 original GPU controls remain. Add20 independent originalF64/full-owner/
status/fault expert controls: six counts(64,65,127,128,129,255) and four poison/
invalid-route controls at65, eachGU/DOWN. Original tolerances and public source
maps remain unchanged. Intended350GPU controls, both complete original18-script
normal/profile/compiler/source/count/owner/erasure/recovery chains and real ISA.
No CPU compilation during GPU timing. Original terminal/clean5%-at-EACH/staged
quality/private-runtime and physical recovery gates precede any installation.
None of these new outcomes is claimed at source implementation time.
