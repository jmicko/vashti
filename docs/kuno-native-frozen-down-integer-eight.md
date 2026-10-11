# Frozen DOWN packed-integer leaf

This default-off source candidate replaces only frozen routed DOWN arithmetic.
It retains all original DASLab IQ4_NL/Q2_0 packed weights, the actual router's
expert and rank identities, full count1..3 waves and the original source
partitions. It is alternate two-residual-Q8 arithmetic, with an eight-lane F32
aggregation tree. It does not claim equivalence to the original64 F32 operator,
whole-model accuracy, speed, installation or release acceptance. The unchanged
original CPU13 tolerance and real target-greedy repair gate remain mandatory.

The actual 48-layer profile contains 39 IQ4_NL DOWN layers and nine Q2_0 layers.
Each expert projects640 input columns to2560 output rows. IQ4_NL retains each
18-byte record's original half factor and32 natural-order signed codebook
values. Q2_0 retains its18-byte64-value record, signed codes−1..2 and common
original half factor across both32-value activation blocks. There is no weight
conversion, reordered expert, model reload, host readback or fallback math.

The quantizer is reused verbatim from `iq_moe_integer.wgsl`: each32-value ACT
block has a ties-even/clamped signed Q8 plane and a second residual Q8 plane,
both with F32 scales. It rejects nonfinite/F32-max inputs and overflowing
intermediate reconstruction. Provenance and MIT notices remain in
`apps/kuno/benchmarks/native-text/src/iq_moe_integer-NOTICE.txt`.

Pinned MIT Strata revision `fb58e0dbc8399662c0e47c76578c6e878b14f6cf`,
`sycl/src/kernels/cuda/iq_kernels.dp.cpp`, `Multi<20>`, `Multi<42>`,
`row_dot_multi` and `native_down_multi_kernel` demonstrate packed payload reuse
and eight-lane IQ4_NL DOWN. This candidate uses Kuno's existing two-residual-Q8
representation, rather than Strata's single-Q8 half scales or fused quantized
SwiGLU. It quantizes the existing original F32 ACT after all five original GU
source dispatches have completed in the same stream.

## Exact leaf ABI

Module: `quant_matvec::frozen_down_integer`. Existing DOWN kernels and default
source selection remain unchanged. The parent owns the separate immutable
selection and actual completed receipts; this leaf never decides policy.

| Constant | Value |
| --- | ---: |
| MAX_COUNT / MAX_ROUTES |3 /30 |
| HEADER_WORDS / FRAME_WORDS |64 /360 |
| LOGICAL_WORDS |10864 |
| SCRATCH_WORDS / OWNED_WORDS |16384 /16384 |
| MAIN_BYTES / HOST_INCREMENT |65536 /65536 |
| INFO_INCREMENT |196608 (`2*48*2048`) |
| RUNTIME_INCREMENT |462848 (`INFO+2*2048+4*MAIN`) |
| MAX_PREPARE_SHARED_BYTES |4004 |
| MAX_PROJECT_SHARED_BYTES |768 |

There are at most30 total route-indexed ACT vectors, rather than30 groups
times three members. Frame`route` begins at`64+route*360`:160 first packed words,
160 residual packed words,20 first-scale words and20 residual-scale words.
Header0..29 contains per-route ready flags; header30..63 is zero. Real
`count*10` routes must all become ready. Inactive frames and10864..16383 remain
zero. Allocate exactly16384 logical words, authenticate the full64KiB root,
and include it in full owning reset/zero-close audits. Meaningful-only scratch
binding or a borrowed subspan is not accepted.

The functions are:

```text
prepare(client, geometry, spans, existing7Handles, scratch, count)
launch(client, geometry, spans, down, existing7Handles, scratch,
       first_slot, slot_count, groups, count)
audit(client, scratch, complete32768ByteFaultOwner, count)
validate_count(count)
validate_geometry(geometry)
validate_limits(limits)
validate_layout(geometry, spans, owner_words7, scratch_words, count, groups)
```

Each uniform is exactly eight public/address u32 words (32 bytes), using the
existing INFO allocator. Count remains immutable runtime data, never a kernel
or pipeline key. Kernel keys contain only stage, original DOWN dtype and row
bytes. Direct and checked execution compilation retain the same raw WGSL.

| Stage | Binding order | Uniform words | Fixed grid / lanes |
| --- | --- | --- | --- |
| Prepare | ACT, STATUS, scratch, uniform | act_first, act_stride, status_first,0,0,0,0,count |1×30×1 /64 |
| Project | original DOWN, MAP, PART, STATUS, scratch, uniform | map_first, first_slot, slot_count, partial_first, partial_stride, status_first, groups, count |320×groups×1 /64 |
| Audit | scratch, complete fault, uniform | count,0,0,0,0,0,0,0 |256×1×1 /64 |

Prepare always launches all30 workgroups, including count1/2. Each workgroup
clears one complete frame and its ready flag, then independently prepares its
real original ACT route only after GU readiness. Group0 exclusively clears
header30..63 and the physical tail. Every scratch word has exactly one reset
writer. All workgroup quantizer cells are initialized explicitly; no implicit
WG-memory initialization is required.

Project retains the five original `(first_slot,slot_count)` pairs:
`(0,72)`, `(72,72)`, `(144,144)`, `(288,144)`, `(432,80)`. `groups` is an
envelope1..30; actual private groups are1..10*count. A uniform guard validates
the full wave, source partition, exact DOWN payload extent, members, per-route
readiness and one member per token before any source payload or shared write.
Original MAP routes address ACT frames and PART rows directly. Eight physical
lanes handle each of eight output rows. Each lane decodes an original encoded
32-value payload once, reuses its signed packed quartets across both planes
and up to three actual members, applies the original F32 factor and plane
scales, then reduces with a fixed4/2/1 tree. Every reduction cell is rewritten,
including unused member columns. Original DOWN status at`80+route` is retained.

Audit inspects all16384 words, checks every real ready flag, inactive frame,
header/tail word and finite nonnegative active scale, and ORs sticky fault8.
It does not clear fault state. Original finite, router, probability, source
fold and all-family acceptance checks remain required and are not replaced.

## Parent obligations

The parent must reserve and precharge every new MAIN, HOST, INFO and runtime
increment before allocation/construction, bind all48 actual layer dtypes, and
authenticate full physical roots, same device/client/stream, original source
identities, count/pending MAP epoch, all actual aliases and immutable policy.
The raw Handle checks are supplementary and do not prove device/allocator
ownership. Device admission includes packed signed-dot4 capability,64 threads,
4004B shared memory, five storage bindings,32B uniform, grid bound320 and the
complete original source-bank binding limits, in addition to64KiB scratch.

The required per-layer order is all five original GU dispatches, one prepare,
five original source DOWN calls replaced by this leaf, one complete scratch
audit, then the existing finite/fold/real target transaction. A successful
completed receipt must bind actual preparation:DOWN:audit counters1:5:1 and
the existing source/router/all-family closure. A requested flag alone cannot
credit execution. Retain old attachment wrappers with false defaults. Reject
staged HOST, private/workgroup tables, shared GU factors, shared DOWN scales,
paired GU, three-member geometry, live-member reduction, and the separate
original-arithmetic four-row DOWN or DOWN stride policies in the first
accepted composition. Although the leaf can validate the four-row metadata,
the initial parent requires plain Frozen DOWN metadata because the integer
kernel has its own eight-row geometry. Plain original GU or the independently
accepted integer GU may compose.

Preparation adds at most76,800B of original ACT reads and64KiB reset writes per
layer at count3; audit adds64KiB reads. Both add one dispatch each. The packed
projection issues320 signed dot4 instructions per row/member across its two
planes instead of640 scalar original multiplications, but packed weight
decoding, scale arithmetic, source memory and enqueue/proof overhead still
matter. These are operation/traffic counts, not predicted speed ratios.

Focused source tests check all48 profiles and original partitions, exact
MAX8 owner spans and charges, independent device limits, actual Naga/SPIR-V
with WG zero initialization disabled, packed-dot instructions, dense used
bindings and no Function arrays, independent pinned-C weight bits at both
record alignments, all30 route projection vectors against a separate F64
quantized oracle, original route coverage and full64KiB reset/audit guards.
These checks cannot establish physical GPU correctness or model accuracy.
Those require unchanged original CPU13/full real repair and matched language
timings on the intended Intel device before any default or release change.

Source validation on the standalone native-text manifest with `direct-device`
and two CPU jobs: all seven new `frozen_down_integer` tests and all17 existing
`iq_moe_integer` decoder/quantizer/prepared/oracle tests pass; strict all-target
Clippy with warnings denied passes. Focused Rust formatting and whitespace
checks pass. No physical GPU operation was performed in this leaf task.
