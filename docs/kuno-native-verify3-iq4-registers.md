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

Implementation is source-only at this checkpoint: focused tests, shader
compilation, strict Clippy and physical GPU acceptance are pending the root
benchmark coordinator's CPU release. No service, public API, frontend,
installation or release changes are included.
