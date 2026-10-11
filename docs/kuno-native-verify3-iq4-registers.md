# Original-F32 Verify3 IQ4 register lookup

This is an explicit default-off native laboratory decoder policy. It changes
only how the original16 signed IQ4 codebook entries are selected. Original
IQ4_NL/IQ4_XS model bytes, F32 activations, scale multiplication, vector dots,
per-lane accumulation, subgroup reduction and complete acceptance remain
unchanged. It does not quantize activations or use integer dot products.
Source tests and shader lowering checks do not establish device speed.

## Source and scope

`next_shared::MatrixSpec::new` always selects `Iq4Decoder::Original`.
`with_iq4_decoder(Iq4Decoder::PackedRegisters)` returns a new immutable spec
and rejects every raw dtype except20 (IQ4_NL) and23 (IQ4_XS). The alternate
policy is supported only by `matrix_verify3`, `matrix_verify3_finite` and the
independent `public_fixture_values` decoder proof. Both original scalar
`matrix`/`matrix_fault` APIs reject a selected alternate spec before dispatch;
there is no silently ignored policy or format fallback.

The decoder policy participates in the existing `MatrixSpec`/`Kernel` hash
and `KernelId::info`. Original and register policies therefore have different
pipeline identities at identical public geometry. Runtime count1..3, offsets,
source identity and input/output contents do not enter the compilation key.
The original kernel names, workgroup/grid, bindings and INFO layouts remain
unchanged. The runtime caller must prove the selected policy through actual
successful IQ4 dispatches and whole completed-target acceptance; a requested
flag or pipeline name alone is not execution evidence. Parent/controller
selection and completed accounting are separate work.

The helper builder replaces only the original IQ4 lookup region when producing
Verify3 or fixture source. `ornith/matrix_vector.wgsl` remains intact, including
its old scalar source and every original packed address/factor expression.
Generic routed experts, token/PLE embeddings, Q5_K and the Q6 vocabulary head
are outside this candidate. For the pinned original DASLab profile,89 matrices
per full traversal are reachable:14 attention IQ4_XS matrices and75 shared
expert matrices (47 IQ4_NL DOWN,15 IQ4_XS gate,13 IQ4_XS up). Shared DOWN is
2560 output rows by640 input columns; shared gate/up are640 by2560. These
counts describe tensor scope, not a performance weight or measured gain.

## Exact lookup

The original GGML b11232 nonlinear codebook is retained under its MIT notice
in `crates/kuno-text/NOTICE`. The implementation reuses the four packed scalar
literals already accepted by `quant_matvec::iq4_value`:

| Codes | Little-endian signed-byte literal |
| --- | --- |
|0..3|`0xbfad9881`|
|4..7|`0xf6eaddcf`|
|8..11|`0x26190d01`|
|12..15|`0x71594535`|

Three scalar threshold selections choose a word. The low two index bits
choose its byte. Exact signed-byte conversion and exact integer-to-F32
conversion produce the original lookup value. The packed nibble caller
already proves every index is in0..15; no new owner or codebook storage is
created. The IQ4_NL18-byte block layout, both nibble halves, IQ4_XS136-byte
block layout and original six-bit signed subscales are untouched.

No MAIN, HOST, runtime/INFO, global/private/workgroup owner, binding, dispatch
or barrier is added. Existing Verify3 finite/tail/full-owner/fault proofs,
all-span checks and whole-model erasure/closure retain their full obligations.
No existing numerical tolerance is changed.

## Required acceptance

Focused tests cover explicit default/type rejection, scalar rejection,
pipeline separation, all16 signed codes and all65,536 mixed nibble quartets,
every committed independent pinned-C IQ4 record including signed zero and
scale boundaries, checked/unchecked WGSL/SPIR-V compilation without Function
arrays, and exact source equality outside the lookup region. Count1..3 retain
the original MAX8 spans, finite last-tile guards, padding and partial-owner
rejections. These are source/host checks; host lookup emulation does not prove
physical shader output.

Before performance selection, run both policies against independent pinned-C
decoder values and original dense controls for counts1..3, widths256/2560,
partial final row groups, nonzero offsets, unaligned original IQ4_NL rows,
full inactive/gap/padding/fault owners and sticky/malformed failures. Preserve
the existing120 dense operator cases and18,112 exact C value proof; add the
candidate rather than replacing the control. Then require unchanged original
CPU13 full vectors, greedy IDs, actual verification/repair, all completed
family/owner closes and physical recovery. Only afterward compare the same
frozen artifact with compiler-idle ordered normal-language timings. Removing
a Function array does not itself prove Intel spills or a speed gain.

## Existing fixed public physical fixtures

The existing compiled-public operator modes now execute additional register
decoder cases without replacing any original control or introducing a new
selector, mode or general inference API. Use the ordinary default binary;
the existing Verify3 registration rejects featured builds, and the original
dispatch guard remains unchanged for IQ fixtures. The root
coordinator retains swap/core-dump protection, exclusive Intel adapter setup
and physical recovery checks. No new Nvidia trial is requested by this change.

| Existing command | Existing fixed environment gate | Original coverage retained | Added register coverage |
| --- | --- | --- | --- |
|`intel iq-quant-probe`|existing IQ public-fixture registration and `KUNO_PUBLIC_RECEIPT`|all original IQ cases,120 shared dense cases and18,112 shared exact C values|4 exact C decoder cases,9,408 values across IQ4_NL/IQ4_XS and byte prefixes0/3|
|`intel verify3-dense-probe`|`KUNO_LAB_VERIFY3_DENSE_PUBLIC_ONLY=1` and `KUNO_PUBLIC_RECEIPT`|60 normal +20 fault cases|12 normal +4 fault cases|
|`intel verify3-finite-probe`|`KUNO_LAB_VERIFY3_FINITE_PUBLIC_ONLY=1` and `KUNO_PUBLIC_RECEIPT`|60 normal +210 fault cases|12 normal +42 fault cases|

Verify3 adds both original IQ4 formats at widths256/2560 and runtime counts1..3
with nine output rows (partial final subgroup), the same exact source views
and full MAX8 owners. Every candidate case independently compares the old
scalar Original GPU control and original decoded-C/f64 projections. Fused
faults cover each count, active input NaN/infinities and logical output prefix,
gap, inactive row and tail. Independent full physical padding, sticky fault
owner and erasure checks remain unchanged.
Each candidate transaction reuses the existing bounded arena after the same
complete reset. Public source and peak private/INFO/I/O bounds are unchanged.

Original report arrays/counts remain intact. `iq-quant-probe` separately emits
`packed_register_exact_c_oracles`, `packed_register_oracle_case_count=4` and
`packed_register_decoded_c_values=9408`. Verify3 separately emits
`packed_register_cases`, `packed_register_format_count=2`,
`packed_register_numeric_cases=12` and `packed_register_fault_cases=4` or42.
Every per-case decoder field records `original` or `packed-registers` only
after the actual selected leaf, complete numerical/fault/source proof and
owner erasure succeed. Receipt creation remains fail-closed and create-new.

The leaf and additive physical-fixture source pass36 focused checks: five
register-decoder tests, seven independent-decoder/probe tests, nine Verify3
fixture tests, ten original shared-dense tests and five original finite tests.
Strict direct-device all-target Clippy with warnings denied, formatting and
whitespace checks also pass. All physical GPU acceptance remains pending at
this checkpoint. No service, public API, frontend, installation or release
changes are included.
