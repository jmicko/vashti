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
