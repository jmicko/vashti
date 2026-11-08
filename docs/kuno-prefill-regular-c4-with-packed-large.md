# Complete regular C4 matrices with packed LARGE operands

Matrix-only source integration of held implementation `e451e687` / `a8b00142`
onto packed LARGE plus hoisted-base checkpoint `2317dbd8`. The complete prior
packed candidate stays retained. This integration has no Cargo/generated-shader,
GPU numerical, whole-model or speed acceptance yet.

Regular32..255-member groups use a distinct `quad_regular` matrix entry with
four FP32 C fragments, eight FP16 A fragments and two FP16 B fragments. Both
K16 halves of each K32 bank are present together; every accumulator retains its
own ascending K16 chronology. This is new shader code, not selection of the
historical C4 kernel. That historical entry lacks whole-layer GU prefix
addressing, so its numerical evidence cannot certify this new entry.

The new entry preserves current dual-regular guards and whole-layer global GU
prefix authentication. Inputs and images stay row-major; INFO14/15 stay0.
Packed LARGE changes only groups>=256, so its layer publication remains
compatible with these unmodified regular segments. DOWN/H uses the original
cohort prefix. WG32/SG16 computes16 members by64 output rows, with unchanged
1024-byte publication scratch. Actual physical X workgroups halve relative to
C2; output coverage, lifecycle and any speed gain still require live acceptance.

The additive pair API is `regular_pair_probe::quad_cases`, reexported as
`cohorts::public_quad_regular_pairs`. It adds exactly eight ordinary/global,
GU/DOWN, finite/poisoned GPU pairs with counts32/33/63/64/65/127/128/255. Global
GU fixtures explicitly require all selected global prefixes to be nonzero and
different from their cohort prefixes. Original per-expert reference, whole
owners/status/fault/sources/tails, erasure, existing host/arena caps and bitwise
comparison stay intact. Every current2117 GPU control remains, so the intended
new total is2125, not yet executed. Existing record fields are unchanged; only
new Q records add explicit quad/full-bank/prefix-check identity fields.

The source provides two additional AppCPU tests, including all16 actual
generated-layout checks through the separate closed four-cell SPIR-V validator.
The standalone extraction of the first real geometry/source-guard test passed
for every regular count32..255; this is no substitute for the second generated
program test. App feature, caller/worker/profile/registration/counter wiring and
native exact-role/grid/layer sealing are separate mandatory parts of the same
complete bundle. Compose them before compilation or any live speed decision.
