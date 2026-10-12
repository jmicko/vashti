# Batched original-Q6 target head candidate

This bounded leaf is based on `08aa57764594356f830210b35149a2c3cf1ad368`.
It is default-unused and changes activation/reduction arithmetic, using the
existing accepted count1 two-residual-Q8 policy. It is not original-F32
identity or a connected target policy. The old count1 sources, public keys and
launchers remain unchanged. No controller, model consumer or device is added.

`q6_integer_matvec::batch` borrows the original Q6 `Spec`, full original weight,
input, dead final-GR NORM, output and sticky fault owners. `PrepareSpans` carries
count/input-first/input-stride/planes-first/planes-stride/fault-first;
`ProjectSpans` carries count/packed-first/planes-first/planes-stride/output-first/
output-stride/fault-first/matrix-first/matrix-rows. Counts1..3 use a distinct
three-component project identity; genuine4 uses a distinct four-component
identity. Count and runtime offsets are INFO, never private cache keys.

Preparation reuses the original32-value quantizer and its exact first/residual
rounding, multiplication/subtraction and packed layout. The new guards use
exponent-bit finite tests; the initial abs(x)<MAX test uses the equivalent
absolute-bit comparison, retaining rejection of ±MAX/NaN/Inf while avoiding
optimized floating NaN comparisons. Original scalar sources are untouched. One
launch prepares all actual RHS rows. For2560 columns each row uses1440 u32
words:640 first codes,640 residual codes,80 first scales and80 residual scales.
Four rows use23040 bytes inside the existing final GR NORM logical extent81920
words (327680 bytes), within its unchanged524288-byte physical owner.
No weight conversion, repack, upload or new private/public/host owner occurs.

The project retains210-byte original blocks,2100-byte rows, original scalar or
aligned16-byte/shift load helpers and the unchanged16 head tiles (15*16384 rows
plus2560 rows). A128-thread group and subgroup16/32 produce the same output-row
grid. Each original16-value code/factor is decoded once and reused across the
actual three/four RHS partials. Each component retains the original signed
quartet dots, first/residual integer sum order, `d*first + rd*second`, factor
multiplication, lane group order and subgroupAdd reduction. This preserves the
count1 alternate policy per RHS, not original F32 activation arithmetic.

Bindings are input/planes/fault/32-byte INFO for prepare and weight/planes/
output/fault/48-byte INFO for project. Each consumes one existing2048-byte INFO
record. Prepare+16 project tiles+full NORM clear fit18 records within the
existing33-record head envelope; all caps and owners remain unchanged. Bounds,
strides, count, finite inputs/intermediates/outputs, inactive rows and complete
matrix complement are checked without reading another tile's active output.
Fault is never cleared by either leaf. Parent still proves full source identity,
actual output.weight geometry, same stream, full physical nonalias roots and
prepared-plane epoch; it retains the final scratch lease through all consumers
and clears the whole NORM owner only after them. Both original fault proofs,
full owner/padding/carry/greedy checks and all-family commit remain mandatory.

The pinned MIT Strata verifier quantizes all final-head rows together and calls
native_mmvq with T rows (`sycl/src/core/verify.cpp`); its wide Q6 kernel shares
aligned loads/decoded weights across RHS (`native_mmvq.dp.cpp`). The existing
native wide-helper notice in `q6_integer_fixtures/NOTICE` remains authoritative.
Our two-residual arithmetic and unchanged weight storage remain separate.

The current original Verify3/4 already reuses decoded weights across RHS. It
scans521472000 original head bytes and635699200 weight positions per row;
three/four rows use1.907/2.543 billion weight/input terms. This candidate reduces
FP32 dot work through packed integer dots; it does not remove three/four weight
scans. No hardware bandwidth or speed claim follows from instruction counts.
Whole-model comparison is required to establish candidate value; these source
checks do not establish an isolated device duration for the head.
Original unchanged numerical tolerances, independent alternate arithmetic and
full protected CPU13/greedy/repair/close/recovery remain required for acceptance.

Source validation: six new CPU/offline tests and ten original Q6 tests pass.
They cover exact per-member quantization/packed order against the independent
C oracle, original helper identity, checked spans, row/tail coverage and all
emitted prepare/project variants through Naga and SPIR-V with workgroup zero
initialization `None`. Project SPIR-V has no Function arrays; this does not
establish driver register allocation or device performance.
