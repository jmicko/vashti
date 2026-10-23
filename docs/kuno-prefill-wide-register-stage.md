# Coupled dense and routed-expert wide register staging

Backend-only optional source2026-10-23,requires complete prefix/padded/direct/
tiny/GR/frontier/expert bundle. Source comparison finds dense and large routed
expert K32 loops both perform next global loads immediately followed by next
SLM stores BEFORE current products. Existing narrow register-prefetch feature
does not apply to these wide kernels. Neither a guaranteed observed memory
stall nor oneMKL exact scheduling is asserted from source alone.

Implement TOGETHER in both real wide producers: guarded next global vectors
remain in actual F16/BF16 local registers across the complete current original
chronological K16 CMMA products; afterward write them into the disjoint next
SLM bank,then the original workgroup barrier and bank switch. Initial staging
unchanged,final K32 block never reads next operands. Real per-thread four vec4
values,not new global owners or additional shared memory. Dense pitch40 zero
padding remains genuine; expert original pitch32 and route/status/fault writers
remain. Only complete512-thread stage-round1 topology selects this source;
original fallback retained for other layouts. No changed precision,products,
read set,barrier count,owners,capacity,erasure or request persistence.

Actual compiled SPIRV must show real StorageBuffer loads before products and
real Workgroup stores after the last product INSIDE the K loop,followed by
its original barrier. Publishing output outside the loop cannot satisfy this.
Add48 such concrete checks (24 dense+24 expert) to unchanged429WGSL/320matrix/
16raw/32compact/24padded validation. Preserve270CPU +three extent/final-block/
source-chronology/actual compiler controls and15 runtime. ALL433 GPU and BOTH
unchanged original18-script chains,native zero-spill code,physical recovery
required. No CPU compile during GPU timing. This expresses an overlap
opportunity; source order alone is not proof of native hardware overlap or gain.

Source only; no build,GPU,speed or installation claim. No API/model/context/
config/frontend/release/reset/reboot change; Vashti:none. Installed binaries
unchanged. Strata5.3515s remains unmet.

## Initial offline audit correction

b37cd753 CPU build succeeds; original controls reach the actual padded-module
check, where the NEW audit reports no product in the first SPIRV loop. The
first-loop assumption was not part of the original matrix contract. Preserve
this failure/logs. Select the UNIQUE loop containing ALL eight original CMMA
products,not simply the first loop; retain real in-loop StorageBuffer-load,
post-product Workgroup-store and following barrier requirements. No GPU/timing
run or shader change for this audit correction. Actual loop counts reported.

The next audit still rejects unique-loop structure before GPU use. Do not guess
at compiler block ordering or relax the eight-product requirement. A test-only
explicit KUNO_TEST_PUBLIC_MATRIX_SPIRV export saves ONLY the static public
compiled module (compiled out of workers); collect actual loop/product metadata
and disassembly before correcting traversal. No speed/GPU claim or rerun.

Actual836a8ae4 offline export confirms ONE loop header115/merge118, eight
products in block160, but zero products lexically before serialized merge118.
Control-flow traversal reaches all eight and next Workgroup-store blocks258/272
then original barrier259. This is serialized block ordering,not a missing matrix
loop. Final audit walks real OpBranch/Conditional/Switch CFG edges,selects the
unique eight-product loop,excludes iteration backedges when proving loads before
products and stores after products,requires a real Workgroup barrier reachable
after EVERY next-stage store. Output stores outside the loop remain excluded.
No shader/maths/owner or GPU change for this audit correction.

## Complete a2da580f validation and one clean

Workera962757951d42aae3e95f5c30bb1e1628156a0dca2a896c51ab1f3b39c1c66c2.
273CPU/15runtime/all429WGSL/320matrix+16raw+32compact+24padded plus48 actual
CFG pipeline layouts,ALL433GPU,BOTH unchanged original18-script chains and
physical recovery PASS. Normal9.802115231/8.495972147s; instrumented
9.891050999/8.593749267s. Actual three dense and three wide expert native
programs SIMD16/128GRF/zero spills; QSA original hashes unchanged zero spills.
Sampled layer24 wideGU5.4106->6.5297ms; no pipeline speedup established.
Kuno11983 resident pairs versus pinned Strata11608; residency is not lower.
Host registered26,215,808,000B,raw banks67,108,864B. These are real allocations.

ONE unchanged clean helper (SHAa0b1b40e5b9ebf00721604c5a2cc1cf6090890a390ff1512e343fdde391f4914)
45/4245/4245,seed0,temp0,4outputs,profiles[] gives first11.238878190s and
repeat8.059392110s. Installed reference10.897223643/8.712129955s. Repeat7.49%
faster,first slower,original5% EACH selection FAILS. Prior complete frontier
clean7.924977071s remains better; Strata5.3515s remains UNMET. No favorable
rerun,staged/install/release. All full and clean public summaries under
`/media/jmicko/faststore/kuno-development/strata-parity-wide-register-stage-cfg-20261023*`.
Final03:43:27UTC same boot: Intel50,688,000B,NVIDIA compute0,faults0,HTTP200,
admission normal. Status read at03:42 was an older snapshot; clean had finished
normally,not a hang. No reset/reboot.

Next grounded source difference: pinned Strata gr_write_norm_rs_kernel holds
its ten original per-thread R values in v[10],then normalizes without rereading
R. Kuno GR_NORM rereads R through a packed-pair mapping and consequently needs
a storageBarrier. Port register retention PLUS subgroup pair packing together;
preserve256-stride sum order and both-level32-lane xor reduction. Strata also
uses256 threads,so a guessed thread-count difference is disproven.
