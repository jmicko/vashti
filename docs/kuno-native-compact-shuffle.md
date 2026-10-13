# Count4 compact expert shuffle reduction

This is an internal native-text candidate based on the accepted compact source
schedule at `fb7b420`. There is no parent selection, API, environment switch,
installed default or consumer change in this leaf implementation. It preserves
the two residual Q8 planes and all existing numerical tolerances.

Distinct `ProjectCompactShuffle` kernel identities and `launch_compact_shuffle`
entries exist only for actual count4 GU and DOWN. Original compact entries and
count1..3 tails remain unchanged. Both entries reuse the complete original
geometry, spans, physical-owner and indirect-view validation, bindings, route
plans, dispatch counts, output/status addresses and full audits.

The caller must enable subgroup support and admit physical subgroup bounds
16 through 32; each shader independently rejects an actual width other than
16 or 32, or a whole-workgroup subgroup count different from `64 / width`.
This also rejects partial subgroups. No local invocation ordering is assumed. With 64 threads, logical
row-within-workgroup is `subgroup_id * (width / 8) + subgroup_lane / 8`, and
logical dot lane is `subgroup_lane % 8`. This covers exactly eight rows and
eight original dot lanes on either admitted width.

Original per-lane decode, packed integer products, scaling and four-member
accumulation remain byte-identical. Unconditional subgroup shuffle-down calls
at strides 4, 2, 1 precede the original `lane < stride` additions. Components
stay independent; lane zero therefore evaluates exactly the original F32
addition tree. This does not use unordered subgroup reductions. Upper lanes'
unused out-of-range shuffle results are never consumed by an addition.

GU keeps its initial shared codebook initialization barrier. Its reduction
arrays and per-group reduction/reuse barriers disappear; the codebook stays
immutable throughout the group loop. DOWN has no remaining shared-memory
reduction or barrier. No buffer owner, scratch extent, allocation cap or
host route fence is added.

Source tests include independent mixed-sign/cancellation bit-tree comparisons
for both subgroup widths, all eight rows and all four RHS members; all packed
GU/DOWN shader variants lower through WGSL and SPIR-V; kernel dimensions,
distinct IDs, guards, unchanged dot prefixes and shared-memory/barrier counts
are checked. At initial handoff these tests have been written but not run:
the root task exclusively owns the compiler cache and GPU. Required physical
gates remain unchanged cold/warm CPU13, complete cleanup and independent
unprofiled code200 timing. No numerical or speed acceptance is claimed yet.

## Root fixed-public selection

The root experiment selects both count4 compact leaves with compile-time
`shuffle::PUBLIC_PROBE`, reported as `compiled_count4_compact_shuffle` by both
public probes. Host admission explicitly requires subgroup support and16..32
bounds before device creation when this compact path is enabled. The rejected
first-plane approximation remains false. No runtime/API selector is added.
The two focused shader-lowering tests passed before selection; full source and
unchanged CPU13/device gates are required for this selected artifact.
