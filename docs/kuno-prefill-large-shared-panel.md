# Packed LARGE workgroup-shared K16 panels

Proposed next complete backend candidate based on frozen ea5597df3806503b9aa79848ba01d2cc907407ac. Current H3 GPU acceptance is still running and must not be edited or interrupted. This candidate has no shader/GPU/model/speed acceptance. The preceding packed/hoist/REG-C4 components remain intact.

## Confirmed difference and scope

Actual Strata oneMKL code uses block2D operands. Current Mesa CMMA lowering still emits repeated per-subgroup global fragment loads; changing pointer vector width alone normalizes back to the same scalar fragments. See B/packed-large-load-message-next-step-20261026.md, including exact compiler sources and standard links. Do not guess a driver large-GRF flag, change backend/driver, or promise speed from source bandwidth arithmetic.

Change only full128-member LARGE prefix acquisition. Preserve the complete packed producers, mixed global X format, regular C4, tails8/16/32, all eight C chains and K16 chronology, genuine owners/grids/limits/status/faults and output layouts. Existing WG256/SG16 computes128 members by128 output rows. Each K16 panel has128x16FP16 weights and128x16FP16 inputs: exactly8192bytes. Sixteen subgroups currently repeat the unique panels four times. Share them once per WG using the existing8KiB fallback scratch; do not add or double-buffer scratch.

## Exact kernel and storage contract

Use ONE SharedMemory<f32>[2048] declaration outside the two branches. Fast staging and original fallback publication borrow it in disjoint paths. Vector<f32,Const<2>>::reinterpret(Vector<f16,Const<4>>) must lower to equal-bit-width OpBitcast only, with no FP conversion/arithmetic on the packed bits. Standard CMMA permits f16 matrices from wider f32 pointees; shared row stride is8 f32 elements for16 halves. CubeCL ReinterpretSlice is unsupported on wgpu and must not be used.

Let K be640/2560, k=0,16,...,K-16, T=tile_index*128, R=CUBE_POS_X*128, I=authentic input_prefix and W=weight_first in halves. Four rounds v=UNIT_POS+round*256 partition1024vec4 source reads. For v<512: r=v/4,q=4*(v%4), source_half=W+((R+r)/8)*(8*K)+(k/16)*128+((R+r)%8)*16+q, shared_f32=(r*16+q)/2. Otherwise u=v-512,m=u/4,q=4*(u%4), source_half=((I+T+m)/16)*(16*K)+(k/16)*256+((I+T+m)%16)*16+q, shared_f32=1024+(m*16+q)/2. Read source_vec4=source_half/4 and store both bitcast f32 elements. Hoist row/parent bases and retain exact K increments; do not reintroduce full row permutations per fragment.

For sg=PLANE_POS, A[j] starts=((sg%4)*32+j*8)*8 f32, j=0..3; B[j] starts=1024+((sg/4)*32+j*16)*8, j=0..1. Keep original A row-major/B column-major declarations and exact eight-C execution order. Every iteration has a WG barrier after staging and a WG barrier after all CMMA use, before any overwrite. No new memory allocation, capability, extension, precision mode, prompt/output persistence or model residency change.

## Workgroup-uniform fast/fallback eligibility

Old plane_all over32 relevant subgroup routes is not sufficient around WG barriers. Each SG must scan the SAME128 immutable routes before branching: eight rounds of map[1540+cohort_prefix+T+round*16+lane], all <map[0]*10. No PLANE_POS dependence in eligibility. All16SG obtain the same plane_all result. Include the extra one-time route work in performance. Prove all earlier header/bounds exits are WG-uniform; no SG may terminate after another enters a barrier.

If any route in the WG is invalid, ALL SG execute the unchanged packed fallback, with its independent8-member finite/status/publication behavior. Valid raw rows must never be written and later cleared as a shortcut. Invalid destinations remain untouched. The map is immutable and source ownership authenticated. No sticky global-fault read that would change fault primitive classification.

## Distinct identity, selection and accounting

Feature strata-large-shared-panels / --large-shared-panels requires the full strata-large-packed-tiles bundle. Native feature kuno-large-shared-panels forwards through wgpu and requires packed LARGE. Keep the existing format1 producers/INFO14 seal; this is a new matrix acquisition program, not a new image format.

New exact prefix identity: ::strata_matrix::cohorts::large_shared::expert_large_shared_cohort::ExpertLargeSharedCohort, tile128 only, same genuine flat-Z prefix grid and WG256. The large_shared launcher delegates tails8/16/32 to the EXISTING packed launcher unchanged; never credit the new identity to those old tail programs. Native grammar must accept the genuine new prefix only with feature+packed format1+original actual owners/header and reject forged/nested/suffixed names, wrongtile/grid/format/phase. Mixed new prefix+old packed tails belong to the same original packed-format LARGE unit.

Use genuine new expert_large_shared_cohort_* prefix counters and exact profile LARGE/cohort/native matrix classification. Preserve generic expert_large_cohort_* totals. Old packed program counters must describe actual old programs; for the new feature, separately validate old packed tails +new shared prefixes ==generic LARGE launches/workgroups. Do not fabricate old-prefix counter credit or silently change old counter meanings. The old feature branch remains unchanged.

New shared compiler inventory is16 prefix configurations (GU/DOWN x8 masks), separately named/validated. All455 WGSL,384 generic,64 old LARGE and64 packed LARGE configurations remain. Original full kernel paths, CPU/runtime/metadata/GPU controls remain independently selectable, with no tolerance/limit relaxation.

## Complete acceptance bundle

Compose before any new live speed decision: kernel/staging/whole-WG eligibility, exact shared-bitcast/stride/barrier/CFG/CMMA proof and negatives, launcher/feature/builder/selection/native layer and format/owner grammar, real counters/profile classification, actual compiler registration, independent host panel oracle and additive paired GPU probes. Work in separate worktrees from ea559; P/H3 stay frozen until current unit terminal. Source-only work during current GPU run; no concurrent Cargo/GPU/root work by agents.

Independent CPU oracle covers every unique source vector/sharedpair once, all role K16 panels/rows/member subtiles/global and cohort prefixes, exact stride/matrix element mapping, aliases/lifetime and fullprefix bounds. Actual generated proof must require one8192B allocation, pure bitcasts, shared stride8, complete and unique stage writes, no global CMMA fast loads, both WG barriers, WG-uniform eligibility/loop, exact original chronological eight-C product and original fallback/store/status/fault semantics. A global-load proof cannot certify shared loads by changing one boolean.

Add actual shared-prefix paired GPU controls against the original accepted per-expert reference: both roles, full-prefix shapes with/nonzero global GU prefix, finite/nonfinite inputs and weights, mixed invalid low/high route halves/subgroups, sticky fault cases as applicable, exact whole output/status/fault/source/tail and erasure checks. Preserve all2125 existing controls and original F64 checks. Use distinct new report records/population. Finish with both complete original model schedules, source/copy/capacity/reuse/ISA/recovery and the original unchanged CLEAN helper/trigger. Inspect actual global/SLM sends, spill/fill/GRF/SLM after compilation; source traffic estimates are not native-code or speed results. Do not roll back individual coupled pieces from isolated speed measurements.

## Source integration handoff

The app/controller wiring is an additive source-only commit atop ea5597df. The
new feature requires `strata-large-packed-tiles`; its explicit builder option
also requires `--large-packed-tiles`. Existing regular C4 selection is unchanged:
`strata_prompt_regular_cohorts.rs` remains byte-identical to the base. The LARGE
controller selects `large_shared::launch` only under the new feature. That
launcher must delegate residual8/16/32 programs to the actual old packed launcher;
old non-shared and unpacked controller branches remain independently available.

Agreed source APIs are `large_shared::launch` with the exact old packed launcher
signature, `large_shared::validate_sources() -> Result<(),Box<dyn Error>>`, and
`large_packed_pair_probe::shared_cases(client,arena) -> Checked<Vec<Value>>`,
reexported as `cohorts::public_large_shared_pairs`. The compiler invokes the new
validator separately from both retained64-config LARGE validators; it does not
replace or add entries to the existing384 generic matrix inventory. Expected
inventories are455 WGSL,384 generic,64 old LARGE,64 packed LARGE,16 shared-prefix
configurations. The existing16 regular-Q configurations remain within384.

The worker adds `shared_large_matrix_checks` and `shared_large_selected=true`,
running the new cases before every old2125 case. The agreed24 new cases are all
**layer1/format1**, preserving global GU input authority:16 base cases cover the
four existing packed count shapes xGU/DOWN xfinite/poison;8 additional mixed
invalid-route cases cover GU/DOWN xoffset0/8/96/104. Prefix0 still uses the real
global GU owner; it is not permission to admit INFO13=0 or an ordinary-owner
format1 alias. Intended total is2149; no new GPU case has executed in this task.
Old raw records, F64 checks, owner/status/fault/tail checks and tolerances remain.

Exact shared identity receives only LARGE tier/cohort/native matrix profiling.
One new AppCPU identity control checks the actual generated Rust type, rejects
forged/nested/appended/unclosed names, and confirms the actual old packed type
retains its own identity and LARGE profile classification. The old timestamp
scope, matrix buckets and regular-Q profile selection are unchanged.

Actual new prefix counters are
`expert_large_shared_cohort_{matrix_launches,prefix_launches,matrix_workgroups}`.
No new shared-tail counter credit is intended. In a full model using this feature,
shared matrix launches equal shared prefixes and generic LARGE prefixes; old
packed prefixes are0; old packed matrix launches equal old packed tails and generic
LARGE tails. Shared plus old packed matrix launches/workgroups must sum to the
authentic generic LARGE totals. Preserve the old feature branch's meanings and
all regular-Q counters. Any later counter-law extension must cover actual direct
layer entry imports as well as `quant_cohort_profile` and `large_wide_counts`;
feature presence alone is not execution evidence.

The wiring task ran only small source checks: five Rust parser passes, TOML/build
selector checks, byte comparison of unchanged REG-Q controller, preservation of
old packed/Q registrations, and `git diff --check`. No Cargo, shared target,
heavy CPU, root, GPU, service, driver, source P or frozen H3 activity occurred.
The actual matrix/oracle/probe and native guard source are owned by separate
agents and must be composed with this wiring before compilation. Their exact
new CPU populations and final whole-source acceptance counts must be established
from the composed source; no large acceptance harness was prepared prematurely.
