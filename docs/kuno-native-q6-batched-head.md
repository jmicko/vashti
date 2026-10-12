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

## Separate paired32-value decode candidate

The next leaf is based on `9ad1ee3aeb8b3ac11eaef1854e8b91cf3695d276`.
It appends distinct `ProjectPairedThree` / `ProjectPairedFour` identities and
uses them only within the existing explicit integer batch head. Old scalar and
old16-value batched source identities remain available unchanged. There is no
new selector, parent connection, owner, capacity, source layout or preparation.

The paired mapping follows pinned MIT Strata `native_mmvq.dp.cpp:1537-1615`:
position `lane & 7`, initial block `lane >> 3`, block increment `width / 8` for
supported subgroup widths16/32. One original ql16-byte payload and one qh16-byte
payload produce both signed16-value vectors. Two scales are read at
`8*(position>>2)+(position&3)` and that offset+4; their global activation groups
are block*16+those offsets. Actual wide views reuse the unchanged aligned
load/shift helper; the checked scalar view reconstructs the same payloads.
The existing `q6_integer_fixtures/NOTICE` retains the MIT notice.

Each16-value contribution keeps the same quartet order, separate first and
residual Q8 integer sums, `d*first + rd*second`, and weight-factor multiplication.
For each RHS the lower contribution is accumulated before the upper contribution
in each block, followed by componentwise `subgroupAdd`. This changes the lane
assignment and F32 accumulation/reduction order; old16-value bit identity is
not claimed. Original CPU13 numerical tolerances and full greedy/repair checks
must pass unchanged. There is no single-Q8 substitution or weakened finite test.

Preparation, bindings, INFO, grid, runtime span/count/subgroup guards, inactive
row behavior, sticky fault, full output-complement audit and whole-NORM erasure
are unchanged. Only scalar/vec/struct registers are added: no Function arrays,
workgroup storage, repacked weights or extra buffers. The source issues half as
many ql/qh payload/helper and half-scale decodes as the old16-value traversal;
the two signed weight scales and all residual dot work remain. At2560 columns,
each row has160 original16-value iterations or80 paired iterations: logical
ql/qh payload calls fall320 to160, binary16-header decodes160 to80, and scalar
metadata byte-helper calls480 to320. Payload helper requests fall5120 to2560
bytes per row before driver CSE/cache behavior; the actual original row remains
2100 bytes and each RHS still performs1280 signed dot4 calls for both passes.
Driver CSE,
register allocation, actual memory transactions and speed are not inferred.
CPU/offline source acceptance passes19 Q6 tests: three new paired checks, six
original batch checks and ten original scalar/load/oracle checks. The new checks
cover6400 independently generated C values, both signed subscales, complete
aligned payload envelopes, exact paired-order packed/natural arithmetic for
counts1..4 and widths16/32, unchanged guards/tails, distinct keys and all34 old/
paired prepare/project sources through Naga/SPIR-V with workgroup initialization
`None` and no project Function arrays. The original operator bound is unchanged.
Strict pure-direct all-target Clippy with warnings denied, scoped formatting
and diff checks pass.
Physical full-model acceptance is pending. No GPU or normal generation result
belongs to this leaf yet.
