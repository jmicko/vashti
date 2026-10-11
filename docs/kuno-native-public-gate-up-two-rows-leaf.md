# Two-row frozen gate/up source leaf

This opt-in kernel packs two complete output rows into the existing 64-thread
workgroup. `Geometry::with_two_row_gate_up()` requires the original frozen
source profile, uniform inactive partition exit, tile 8, hidden 2560 and FF 640.
It rejects private/workgroup tables and GU factor staging in both construction
orders. Four-row down and down-scale staging remain independent choices.

The public gate grid changes 640→320; down, storage owners, INFO, map, real route
ranks, RHS 8 capacity and full scratch spans remain unchanged. Physical lane
`p` serves row `p/32`, retaining original virtual lanes `p%32` and `p%32+32`.
Each accumulator visits the same columns in the same order. Gate and up retain
separate low/high arrays, then perform the exact original stride 32 addition in
registers before shared strides 16,8,4,2,1. Stable SiLU and the original gate/up
product, active-member stores and feature 0 status write remain unchanged.

The exit tests the whole pair using public cube coordinates and the existing
shared group/partition predicate; there is no per-row early return. Two 512-F32
sum arrays still use 4096B shared memory and initialize every cell before the
uniform fence. Reduction requires six dynamic barriers per cube, versus seven
for the original row. Live accumulator state increases 16→32 F32 per thread and
weight decode work per thread doubles; throughput must be measured.

The parent must select `grouped_gate_up_frozen_two_rows` atomically with the
paired Geometry grid. Ordinary entrypoints must never consume that grid. This
source kernel leaf has no GPU acceptance claim.
Focused CPU checks cover all 48 layer geometries, all four original IQ GU
families, exact virtual 64-lane dots and SiLU products, Checked/Unchecked shader
lowering, actual initialization coverage, uniform barriers and runtime SPIR-V
with workgroup zero-initialization disabled. Existing original fixtures remain.

## Internal public probe selection

Both fixed public probes accept `KUNO_LAB_FROZEN_GU_ROWS=two-public`; absence
selects the original one-row gate/up kernel. The canonical value requires
`KUNO_LAB_TARGET_SOURCES=frozen-host-public`,
`KUNO_LAB_FROZEN_PARTITION_EXIT=1` and the existing protected original request
scope. The speculative probe also requires its original warm replay. Table
placement and GU-factor choices are rejected before source imports; four-row
down and shared down scales may compose independently. No production API or
default inference behavior changes.

The choice is fixed in each source layer's Geometry and selects
`GateKernel::TwoRows`, the named two-row entrypoint and its paired grid together.
The attachment validates the same flag contract before importing any pages.
Actual receipts use `frozen_host_two_row_gate_up`, read from attached source
geometry through `native_frozen_source_two_row_gate_up` after the existing
healthy completed native projection/owner proof. Enabled selection additionally
requires a nonzero accepted source-dispatch counter. Receipts are captured before
close. The speculative cold replay reports false; the attached warm replay must
match the explicit request. A selected flag alone is insufficient for true.

The protected launcher remains a separate parent integration. This source
wiring does not release, deploy or execute an inference trial.
