# Complete N128 small-expert temporal carry

Backend-only candidate composed after the complete local-prefetch acceptance.
No compilation, numerical execution, timing or installation claim yet.

Initial source a7fd8ded fails compilation because the new shared module did not
explicitly import `half::f16`; its parent scope does not export that concrete
type. Preserve the failed full-build log. Add the explicit import and rebuild
the same complete candidate. No GPU test has started.

The accepted N128/four-C small macro and cohort consumers have zero spills but
are slower than the earlier N64/two-C carried consumers. Their K16 arithmetic
uses only current-iteration operands. This candidate combines the N128 owner
and publication geometry with genuine chronological K32 loop-backedge carry
for BOTH small paths. Large eight-C projections and experts remain the complete
accepted local-prefetch implementation. All earlier variants remain selectable.

`strata-small-local-carry` / `--small-local-carry` requires the whole
`strata-local-prefetch` bundle. Both actual production selectors choose an
explicit compile-time arithmetic parameter. It uses the retained, unchanged
four-C `native_wide::balanced_wide32::compute`: load bank0 before the loop,
load bank1 before current products, consume K16 in original order, refill
bank0 at K+32 only when another iteration exists, then consume bank1.
Both admitted roles have K divisible by32. No final speculative operand read.
That arithmetic spilled in the earlier large wrapper; its register pressure
in these small wrappers remains unproven.

Original explicit `try_expert`, cohort `launch`, compiler definitions and all
1747 old numerical controls always select carry=false. The production selectors
use the new feature. Compile-time specialization participates in the genuine
CubeCL kernel identity; no runtime scalar, descriptor, owner, group status,
packet word, native role name, guard or physical grid changes. Existing native
roles remain authentic for both arithmetic specializations. No vendor runtime
extension is required. Feature-off rejects a requested carried host launch.

The same initialized padded inputs, N128 grids, four C outputs, independent
eight-member fallback publication, GU checker, ordered DOWN, six owners,
cohort packet, source/release phases and erasure remain. Actual new subcounters
identify carried macro32/64 and GU/DOWN cohort launches, beneath the existing
N128 family counters. No older carry/preload counter receives false credit.
Old tile8 is unchanged. No new request state, GPU allocation or persistence.

Acceptance must retain all421 AppCPU and 1747 GPU controls. One additional CPU
test compiles all48 genuine new small layouts and checks the actual initial,
current and carried operand origins, distinct row banks, chronological four C
chains, bindings, INFO/typed-vec4 stride units, WG/SG, publication and negative
mutations. The original local88 and large carried40 layout controls remain.
Planned full422 / feature-off421 AppCPU; unchanged18/86 runtime and8+9 metadata
and10 native controls. Add116 paired old/new/F64 complete-owner cases and4
mixed-slot cases: planned1867 fresh GPU controls. Existing numerical tolerances,
invalid-route sentinel and nonfinite semantics stay unchanged.

Compose the entire feature, production selection, compiler checks and both
paired consumers before live tests. Then require both original complete model/
compiler/counter/capacity/erasure/recovery chains, actual ISA/spills and the same
original first/repeated CLEAN gate. Do not credit source spelling, estimated
cycles or an isolated microbenchmark as prompt-speed acceptance. No API,
Vashti, installed service, driver, boot or release change is authorized here.
