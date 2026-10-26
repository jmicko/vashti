# Independent packed LARGE operand-layout audit

Source-only audit of the proposed packed LARGE bundle against diagnostic base
`f1cc44457aa02f3940792523c4d9792f483ced2c`. This audit does not change the
accepted row-major engine, global capacities, owner grammar, or specifications.
It found no address mismatch in the proposed permutation. It makes no GPU,
floating-point, quantized decoder, full-model, compiler, or speed claim.

The additive standalone Rust test is
`apps/kuno/benchmarks/native-text/src/strata_large_tile_layout_oracle.rs`.
Run it beside the draft `strata_large_tile_layout.rs` with:

```sh
rustc --edition=2024 --test -O \
  apps/kuno/benchmarks/native-text/src/strata_large_tile_layout_oracle.rs \
  -o /tmp/kuno-large-layout-oracle
/tmp/kuno-large-layout-oracle --test-threads=1
```

The independent reference visits row tiles, reduction tiles, rows inside each
tile, and contiguous reduction columns. It does not use the candidate address
formula to construct expected physical order. Seven additive tests plus the
included draft helper's original test passed on 2026-10-26 UTC.

Coverage includes exhaustive element bijections/inverses for B8/B16 and reduction
extents16/32/640/2560; exact 2/4/8-half vector alignment and ordering; all eight
full GU/DOWN image slots without gaps or overlaps; every original A/B stride16
fragment; a mixed count512,5120-route global X owner spanning small/regular/LARGE
boundaries and partial Pad32 groups; exact zero gutters; activation/H cohort
prefixes and untouched owner suffix; the actual flattened activation dispatch's
permuted zero-loop schedule (each gutter word exactly once); actual direct8
producer vector-address formula; nonzero-parent prefix equivalence and
checked-overflow negatives. Public synthetic bit patterns include high bits and
are compared without float conversion.

Implementation constraints that the complete bundle must retain:

- A dequant image element is `vec4<u32>`: eight FP16 halves. A matrix element
  is `vec4<f16>`: four halves. Global X/H stores are `u32`: two halves. The
  corresponding divisors are8,4,2; confusing them silently changes the image.
- GU weights still alternate gate/up row identities; layout packing must not
  change decoder source row or projection. Weight slot/role bases are outside
  the per-role permutation.
- GU inputs use the global X group prefix. DOWN inputs and H publication use
  the cohort prefix. Those prefixes can differ for the same group.
- Applying B16 packing to a global row agrees with a per-group permutation
  only when the parent prefix is a multiple of16. Original Pad32 prefixes
  guarantee this. The test includes a nonaligned counterexample; keep all
  original prefix%32 guards and the native layer format seal.
- Only groups with at least256 members are packed in global X. Smaller group
  segments remain byte-identical row-major, including their Pad32 gutter.
- Cooperative A remains RowMajor and B remains ColMajor. Both have a leading
  dimension of16 under this layout; the original K16 accumulation chronology
  and eight C chains are separate unchanged numerical requirements.

These CPU checks cannot prove that a generated producer emits the expected
layout or that Mesa lowers the intended cooperative load efficiently. Actual
paired GPU full-owner producer/consumer checks, generated SPIR-V and native
format/ownership controls, original numerical controls, both model chains,
original clean speed comparison, and physical recovery remain required.
