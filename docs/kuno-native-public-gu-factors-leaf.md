# Public frozen gate/up subgroup-factor leaf

Source-only candidate based on agreed integration
`ce78d391ccfa90bb8ed8d7db996c30011c2aa150`. No environment or launcher selector
is added by this leaf. GPU correctness and throughput are untested.

`Geometry::with_shared_gate_up_factors()` requires the original frozen source
geometry and explicit inactive-partition exit. The policy composes independently
with either module-private or workgroup public codebooks and with shared down
scales. Private and workgroup codebooks remain mutually exclusive. Source hooks
are `Sources::new_with_shared_gate_up_factors(...)`, the combined
`Sources::new_with_policy(...)`, `Sources::shared_gate_up_factors()` and
`Workspace::frozen_host_source_shared_gate_up_factors()`.

Each operative CubeCL GU workgroup handles one output feature and up to eight
route members; tile8 is the RHS member tile, not eight output features. An active
workgroup stages its exact gate factors followed by its exact up factors, with
one lane-strided initialization and one additional unconditional barrier before
the original column/member loops. Inactive groups return uniformly before
initialization. Frozen D=2560 fixes the factor array to 320 F32 cells for IQ2_S
(16-value subgroups) or 160 for IQ3_XXS/IQ3_S (32-value subgroups). The one IQ4_XS
layer compiles the original path without factor storage or an extra barrier.

Factors retain the original half conversion and multiplication order:
IQ2_S `d * (0.5 + scale) * 0.25`, IQ3_XXS
`d * (0.5 + word_high_nibble) * 0.5`, IQ3_S `d * (1 + 2 * scale)`.
Original magnitude/code and sign decoding, weight multiplication order,
column/member accumulation, 64-lane reduction and SiLU remain unchanged. No
activation quantization, weight expansion, regrouping or explicit FMA is added.
Every unselected decoder and entrypoint retains its original path.

GU shared-memory storage, including original two 512-cell reductions:

| Profile | Original/private tables plus factors | Workgroup tables plus factors |
| --- | ---: | ---: |
| IQ2_S | 5,376 bytes | 13,568 bytes |
| IQ3_XXS | 4,736 bytes | 5,760 bytes |
| IQ3_S | 4,736 bytes | 6,784 bytes |
| IQ4_XS | 4,096 bytes | 4,096 bytes |

The GU and down kernels have separate workgroup storage. Source bindings, MAIN
owners, INFO spans, source grid, runtime maps and original model owners are not
expanded. Existing source metadata reservations remain in force. All 48 source
geometries are reconstructed and compared before immutable attachment.

Exact new GU entrypoints preserve plain, private and workgroup codebook policies.
The guarded compiler recognizes only their exact public names. Parent integration
must extend its actual kernel-policy mapping, factory, parser and protected
launcher, then compare and record the completed attachment policy against the
requested mode in both CPU13 and language receipts. A true geometry flag alone
does not prove that runtime selected the corresponding entrypoint.

CPU source acceptance passes six focused direct-device tests: checked and
unchecked WGSL plus SPIR-V with workgroup zero initialization disabled, exact
lane-strided factor coverage including the gate/up boundary, all-path barrier
dominance, three adversarial barrier/immutability cases, exact independent pinned-C
weights and original 64-lane dots, unchanged IQ4_XS/down/finite/reduce paths, and
all 48 geometry/prerequisite/policy composition and charge checks. Test compilation
took 24.45 seconds and the six checks took 20.20 seconds. Strict direct-device
main-binary plus tests Clippy with warnings denied passes (25.62 seconds), as do
formatting and diff checks. A peer static production audit found no numerical,
guard, source-policy or dispatch defect.

Re-gate full original CPU13 vectors/greedies/repair and complete owning close
before same-artifact normal-language timing. Source checks do not establish
physical arithmetic or speed. The extra factor writes, barrier and SLM residency
may outweigh repeated scalar factor work; compiler/driver reuse is not measured.
