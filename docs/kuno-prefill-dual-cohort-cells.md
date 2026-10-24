# Two-accumulator cohort cells

Source continuation after accepted regular-cohort runtime `bac6da32` and
worker `42e3bc4e`. This whole new variant is composed but NOT yet compiled,
GPU-validated or timed. It is not installed. Strata parity remains unmet.
Required Vashti/API changes: none.

## Transfer audit before selecting the next kernel

The accepted normal L24/main4096 receipt actually witnesses/selects/submits
rolling execution:665 compute nodes/878 total nodes and one genuine admitted
cohort recording. In `strata_graph.rs`, normal `profile_active()` is false
for `prefill-host-layer`; `Planner::for_lane(...,profiling,0)` therefore uses
nonserialized MAIN/COPY planning. The original adopted `KunoTransferQueue`
identity/authority is passed to real `Commands::begin_rolling`, not invented.

Both original whole raw banks are prefetched; retiring a fully enqueued unit
permits unit+2's copies to be enqueued. The real dequant DOWN image has
`native_release_after=true`. Original read/write range hazards and timeline
waits make a raw overwrite wait on dequantization, rather than its following
GU/DOWN products. `strata_schedule` has actual same/different-bank, compacted
and uncompacted CPU controls for these dependencies. Real COPY submits use
the adopted transfer authority. Original timeline signals, host-upload header,
all-queue terminal join, recording owners and completion fences remain.

The original timestamp sampler deliberately calls the planner with
`serialize=true`, disables rolling, and stamps individual nodes. Its28.416ms
copy sum cannot establish absence of overlap in the normal schedule. This
source/receipt audit proves selection and allowed overlap, NOT measured DMA
concurrency or physical bandwidth. Do not add another transfer queue or change
hazards merely from the serial sample. The source audit did not find a missing
overlap mechanism. Native library GEMM internals in pinned Strata remain opaque.

## Complete optional source policy

`strata-small-dual-cells` / `--small-dual-cells` requires the ENTIRE accepted
regular-cohort/HOST/granular bundle. Keep every earlier complete variant.
Compose BOTH regular32..255-member and original9..31-member cohort16 consumers.
Original tile8, >=256-member large path and old four-C consumers stay intact.

Each SG16 produces16 members x16 output rows using two F32 C fragments.
WG32/two subgroups produces M16/N32 rather than the preceding M16/N64.
New exact generated types live in `cohorts::dual_regular` and
`cohorts::dual_small`; native role classification, physical rows/32 grids,
full INFO/phase/source/stream/owner checks, actual counters and compiler proofs
change together. New roles cannot fall into old loose substring grammar.

Four A and two B fragments implement guarded K32 carry. Initialize A/B(K0),
load A/B(K+16), consume both C at K, refill ONLY released A/B(K+32) behind the
original final-iteration guard, then consume both C at K+16. Every individual
cell retains ascending K16 accumulation. No precision, FMA/round point,
valid-route, finite/fault/status/pending2 or eight-member publication change.
The real publication scratch is still256 F32 words/1024 bytes: two subgroups
each use128 words, reused by both C fragments. Do not advertise512 bytes.

Original packed Pad16 producers, image layout, both raw banks,75MiB images,
80MiB working owner,20MiB X/5MiB H, source residency and every existing capacity
remain. No new runtime owner, decoded model cache or persisted request data.
Physical WGs double for the affected consumer cells; useful output arithmetic
and weight-A reads stay fixed, input-B reads double. Less accumulator liveness
and more independent work are hypotheses, not measured occupancy or a promised
speed gain; both preceding programs already compile without spilling.

## Acceptance required before timing or claims

Retain all442 existing selected AppCPU controls and same-source442 feature-off
controls. Add four lattice/carry/producer-selection/profile controls. Original
four-C compiler validation stays strict through its original entry; a separate
closed two-C entry proves9 total operand loads,6 stores,4 products, exact
StorageBuffer/vector strides and K-loop reaching/accumulator dependencies.
Every original module/resource/negative control remains. Add32 actual layouts
(two roles x two families x eight compiler configurations), WG32/SG16/1024B.

Retain18/96 runtime suites; add four exact genuine new-grid/owner/phase and
forgery controls. Existing GPU families remain intact. Add12 regular and4 mixed
small paired cases using the SAME original complete-output/status/fault/source
oracles and bounded erasing arena. New cases explicitly identify the actual
dual consumer; old cases still execute their real old consumer. Planned fresh
ALL2027 = original2011 plus16. No receipt relabeling or tolerance relaxation.
Both complete original model/counter/compiler/erasure/recovery chains and the
unchanged original CLEAN gate follow complete source and numerical acceptance.

No speed-based isolated rollback. Keep this complete variant and fix any source,
compiler, correctness or owner defect before drawing a performance conclusion.
No install/release/API/Vashti/canonical/driver/reset/reboot change is authorized.
