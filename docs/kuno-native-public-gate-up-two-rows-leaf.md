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
leaf changes no resident/probe/launcher policy and has no GPU acceptance claim.
Focused CPU checks cover all 48 layer geometries, all four original IQ GU
families, exact virtual 64-lane dots and SiLU products, Checked/Unchecked shader
lowering, actual initialization coverage, uniform barriers and runtime SPIR-V
with workgroup zero-initialization disabled. Existing original fixtures remain.
