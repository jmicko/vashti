# Opt-in paired-cell prompt attention — 2026-10-20

This is a CPU-validated source candidate, not an installed change or GPU result.
Feature `strata-qsa-paired` applies only to `PromptAttention` and
`PromptAttentionFault`, after their existing scalar-state/vector-query expansion.
Default inference, decode, MTP, selection and the unchanged scalar golden source
are untouched. There is no private request flag or new API.

## Actual compiler evidence

The fixed-public receipt and compiler dump are under
`/media/jmicko/faststore/kuno-development/strata-parity-native-wide-20261020-operators`.
Receipt source `9cf2236bd62abcf77f996e56f4a580d3e26d5abf`, worker SHA256
`a7ac11f1832322a589f6fac5f55711d556ad4ff1ecf27ea775e1356f7cf9390e`;
all103 actual controls pass with physical recovery. Dump SHA256:
`c1ccaedecfa339c82422e85d0182a8a866e48df284c267f49d2648d44ed2d00e`.
The protected helper enables only `INTEL_DEBUG=cs` on its compiled-public run.

The attention fingerprint is WG256, SG32,15,620 bytes shared: query3072 F32
values (or768 vec4), probability768 F32,64 row IDs and one broken atomic. The
prompt authoritative-cache rewrite removes the candidate binding; fault and
non-fault forms have distinct binding indices. In the actual dump:

| Mesa source hash | Source role | Instructions | Loops | Sends | Spills:fills |
| --- | --- | ---: | ---: | ---: | ---: |
| `708d73cdf3b05dea` | unchanged scalar prompt attention |2002|2|180|0:0|
| `f2d86f271efdc26b` | specialized typed-vec4 prompt attention |1885|2|146|0:0|
| `eb56181812ab4d50` | unchanged scalar prompt attention/fault |1599|4|118|0:0|
| `3ed030e7b78501f6` | specialized typed-vec4 prompt attention/fault |1988|2|146|0:0|

The roles are attributed from the pinned generated source, distinct original
binding layouts and exact shared declaration fingerprint in the NIR. The dump
does not retain a per-pipeline named SPIR-V file, so these are not independently
recomputed Mesa hashes. Instruction/cycle estimates are compiler diagnostics,
not hardware timings. All report the actual pre-Xe3 ANV128-GRF mode.

The35:3 spill cases (`40bcba630ae61f83`, `8e4fc96d2bb43b1b`) instead match
QSA selection's WG1024/shared32,912-byte integer scratch:8192 keys,33 subgroup
scan words and three scalar words. The26:52/53:122/28:56 examples match GDN
recurrence WG128/shared1540 bytes (three128-F32 arrays plus broken atomic),
not attention. No unsupported register-count environment setting is proposed.

## Connected change

One original logical32-lane subgroup scores cells `warp,warp+8,...`. The new
loop handles `(pair,pair+8)` together, advancing `pair` by16. It loads both
original eight-value K fragments, then reads each head's two query vec4 values
once and uses them for both dots. Each cell keeps its original eight products,
left-to-right additions, XOR16/8/4/2/1 reduction and division by16.

For a full64-cell workgroup, this reduces expressed repeated query shared reads
from786,432 to393,216 bytes. These are source-level byte counts, not measured
transactions or a speed prediction. The selected pinned SYCL default does not
use this transformation. Its separate experimental pre75 NC=2 path demonstrates
the same query reuse principle; this candidate keeps our original SIMD32 route
and does not copy that path's other reduction, probability or architecture
choices.

Both cell IDs are in0..63 exactly once. Inactive tail cells receive the same
negative score sentinel; causal IDs are checked before any cache read. Invalid
visible IDs, nonfinite K and nonfinite scores still set the original broken
atomic. Only original non-fault variants write the original per-cell status.
The entire source before QK and after its ending barrier is unchanged, including
query offsets, physical owner indexing, softmax, chronological PV FMAs, output
padding, full closing status/fault publication and all four barriers. There is
no new state, owner, scope or authority.

## Validation and remaining gates

`test-prompt-qsa-specialize.py --deps-dir PATH --paired` directly compiles the
actual source generator and existing Naga/SPIR-V writer using prebuilt Rust
dependencies; no Cargo target, model, GPU or service is used. The ten CPU checks
cover original specialization plus paired source lowering, all0..64 tail
geometries, exact per-dot/XOR source order, unchanged prefix/softmax/PV/closing,
all four barriers, unchanged twelve explicit PV FMAs, and NoContraction on every
actual paired SPIR-V FMul. The same tests also run without the feature to prove
the original production hook stays selected by default.

Before selection, require the existing complete GPU inventory, including all
nine strict bitwise QSA controls and all three independent F64 controls with
unchanged bounds/full owners/tails/faults. Then complete original short/tail,
long, thinking/tools, semantic-repeat, privacy/erasure and physical recovery
checks. Inspect actual generated ISA before judging speed: the extra live K
fragment can increase pressure and spill despite halving query reuse work.
Only then compare clean matched prompt timing. No performance gain is claimed
from CPU validation or the compiler's estimated cycle counts.
