# Authenticated padded medium inputs and pure-global matrix products

Internal candidate after completed single-K source a9d31cc7. The user returned
GPU access and authorized continued speed work. No public API, Vashti, installed
configuration, driver or reboot change. Preserve every prior complete variant.

`strata-padded-global` / `--padded-global` requires the complete single-K bundle
and propagates `kuno-padded-global` through the native runtime. BOTH actual
Gather and validating SwiGLU producers initialize every extra FP16 input word
to the next32-row boundary for real counts32..255, using their original real
dispatch grids. Counts below32 retain original8/16 padding; larger counts are
unchanged. Every padding extent is checked against the actual output view.
Real rows, route count, pending-status/finite rules and output sentinels remain
unchanged. The existing4096-row private input owner already holds these rows;
this grants no permission based merely on unused capacity or initial clearing.

The exact new producer identities end in `Pad32`. They normalize only for
existing role validation; native grammar records initialization authority
separately. New closed `native::padded_global` matrix identities require BOTH
the authentic preceding producer and its complete rounded input extent.
Old Gather or old checked activation cannot grant this authority. Forged
names, changed owners/grids/info or undersized inputs remain rejected. GU
still requires the original validating activation before ordered DOWN.
Padding authority is discarded at the original group boundary. A same-layout
legacy Group comparison must show the boolean fits existing metadata padding.

New consumers keep original M32/M64,N64,SG16,WG32/64 and original scalar ABI,
real-count grids and output ownership. All eight accumulators use ONE pure
global K16 loop, with no per-step input branches, input stage or handoffs.
The original independent8-member publication and checked GU/ordered DOWN
consumers remain mandatory. Actual publication scratch is1024/2048B; original
full-walk2048/4096B remains compiled separately. Original cohort16/tile8/wide
paths retain their existing arithmetic. No new private GPU buffers, phases,
tape capacities or retention policies.

All prior398 AppCPU and1223 GPU controls remain selected. Add actual emitted
new-module proof, new producer source/identity control, and real profile-name
classification =>401 AppCPU. Add two native full-role/forgery tests including
old-producer rejection, missing H padding and short ownership =>69 full runtime
with the original18 feature-off,8+9 metadata and10 native controls retained.
The new emitted-module proof retains exact real geometry/operand/AtomicOr/
publication/SSA negatives; it requires a single pure-global eight-product
loop. Original shared-stage negatives still run against the retained module.
Its48 layouts consist of32 new macro layouts plus16 retained cohort layouts.

Add64 genuine new/reference full-owner/F64 cases using explicitly zero-padded
public input fixtures, preserving real-only oracle input and all invalid-route/
NaN/sticky-fault/sentinel checks. Add18 actual old/new Gather/checked-activation
GPU pairs at counts32,33,63,64,65,127,128,129,255, checking every padded word,
real bit, full status/fault/source owner and poisoned tail after reuse/erasure.
Total planned GPU controls1305. Compile and validate the WHOLE bundle before
speed comparison; no isolated component rollback. No acceptance/timing claim
until the exact candidate passes and physically recovers.
