# Packed LARGE prefix: hoisted operand bases

Source-only follow-up to packed candidate `5d7a2194`. The complete preceding
candidate and its frozen acceptance artifacts remain separate and unchanged.
This follow-up has no GPU, generated-program, model, or speed acceptance yet.

The packed prefix computes its two input and four weight row/parent bases once
before the chronological K16 loop. Each iteration then advances input addresses
by64 `vec4<f16>` elements and weight addresses by32. This follows the existing
16x16 input and8x16 weight tile sizes; no format, producer, owner or memory size
changes. Each fragment retains its separate row-block base. Load/DPAS order,
all eight C chains, checked fallback loader, residual paths, publishers, metadata
and the closed generated-SPIR-V validator stay unchanged.

The existing standalone CPU layout oracle now has one additive recurrence test
against independent tile enumeration, covering every K16 position on both role
axes, all eight image slots, nonzero global/cohort prefixes, partial/full prefix
shapes, and exact final-fragment owner bounds. It also checks that the real fast
loop contains only the intended vector advances and no row permutation calls.
All nine standalone CPU controls passed. This establishes address equivalence;
actual generated shaders, register/lowering evidence, original GPU controls,
model chains and clean timing remain pending. No compiler or speed claim follows
from the reduced source-level address computation.
