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
