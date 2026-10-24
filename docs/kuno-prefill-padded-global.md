# Authenticated padded medium inputs and pure-global matrix products

## Completed acceptance (2026-10-24 UTC)

Source `1d678407b298f33bdbd00043b2d875606515cd45`, worker
`d5ab4b7e2e3871e006bb20c5186d6ce192c6b51691d56884d620cc18e53ab500`:
401/398 AppCPU,18/69 runtime,8+9 metadata,10 native, all1305 GPU controls
and both original18-script full-model/recovery chains pass. All1223 preceding
GPU controls remain. New32 macro layouts plus16 retained cohort layouts pass
actual SPIR-V/CFG/provenance/negative validation. Actual publication scratch
is1024/2048B for the new macro32/64, with no input stage. All12 compiled old/
new macro programs have zero spills/fills; the complete207-program ISA export
and original two QSA hashes are verified. Actual new producer counters equal
the full-macro semantic counters in both full-model modes.

Normal warm/first/repeated:4.741365704/12.924652607/10.147959844s;
diagnostic4.721734699/12.925523716/10.252907898s, same4245-token fixture.
Normal/diagnostic load81.202777476/80.493126585s. This is a modest gain from
single-K normal10.519329404s, but does not beat local-route9.360428477s,
best CLEAN8.394039699s, or Strata CLEAN5.3515s. No new CLEAN triggered.
Keep all complete variants; no isolated rollback or installation selection.
Diagnostic macro236 calls34.292184->25.875936ms, cohort278 calls19.910562ms
(essentially unchanged). Other sampled families are unchanged. Intervals may
overlap; do not sum them as wall time. Next investigate a coupled smaller
per-subgroup output cell and more subgroups for BOTH macro/cohort products.

Artifacts under `/media/jmicko/faststore/kuno-development`:
`padded-global-app-cpu-v1-20261024`, `padded-global-off-app-cpu-v1-20261024`,
`padded-global-runtime-cpu-v1-safe-20261024.json`, original root build receipt/
freeze, `padded-global-v3-preflight-safe-20261024.json`, V3 acceptance freeze,
`strata-parity-padded-global-v3-20261024-expert-cohorts-public-summary.json`,
`padded-global-v3-{profile,normal}-safe-phases-20261024.json`,
`padded-global-v3-complete-isa-export-20261024.json`, complete driver log and
`padded-global-v3-final-state-20261024.json`. V3 exits0. All failed artifacts
remain: first wrapper matched the suffix of401-summary as another1-test result;
V2 corrected this but retained three obsolete runtime-difference counts. Exact
V3 read-only preflight validates all prerequisites before GPU launch. Its first
shell attempt correctly failed the swap/core protection check; protected audit
then passed. No CPU compilation/test rerun for the corrected V3 acceptance.

07:50:59UTC recovery: Intel50,688,000B, NVIDIA compute0/desktop656MiB, all
five kernel faults0, all three services HTTP200, admission inactive/unpaused/
no recovery/waiting0/completed230. No installed service, API, Vashti, reset,
driver or reboot changes. Goal remains paused in controller and unmet; user
authorized continued manual work. Bounded refresher remains within its4h limit.

Read-only reference recheck: existing matched benchmark uses4096+148 chunks,
identical to Kuno; this was already recorded in the quant-producer note.
Existing Xe2/128-GRF versus Xe3/256-GRF research also remains unchanged.
Do not spend another benchmark or driver update on either assumption.

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
