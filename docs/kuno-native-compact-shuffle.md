# Count4 compact expert shuffle reduction

This is an internal native-text candidate based on the accepted compact source
schedule at `fb7b420`. Root explicitly selected it for the fixed-public
laboratory probes below and has now deselected it after neutral timing. There is no API, environment switch,
installed default or consumer change. It preserves
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
are checked. Root has now run the complete1343-test suite and strict Clippy,
and the unchanged full cold/warm CPU13 plus cleanup have passed. This supersedes
the initial source-only leaf handoff. Independent unprofiled code200 timing is
required separately from numerical acceptance; see the results below.

## Root fixed-public selection

The root experiment selects both count4 compact leaves with compile-time
`shuffle::PUBLIC_PROBE`, reported as `compiled_count4_compact_shuffle` by both
public probes. Host admission explicitly requires subgroup support and16..32
bounds before device creation when this compact path is enabled. The rejected
first-plane approximation remains false. No runtime/API selector is added.
The two focused shader-lowering tests passed before selection; full source and
unchanged CPU13/device gates are required for this selected artifact.

## Physical numerical acceptance (v196)

Root artifact from `b9ee1bef78a9f4f7eab58dd940455b364dabf1c2`, SHA256
`81cd9d54491da08623ac03454476d6cc2985b5161fa21a97d160f7b9f954cabc`,
passes the original full cold/warm CPU13 vectors, carries, greedies, real repair
and owner closure on the B70. Receipt flags confirm shuffletrue and
primary-planefalse. Warm compaction dispatches are1440 GU,1440 DOWN and288
audits. The full source suite passes1343 tests with one ignored, and strict
direct-device Clippy/fmt pass. All services, admission, actualGPU baselines and
classified kernel faults recover after the trial. Independent normal code200
measured10.955189006 continuations/s:199 continuations in18.164907962s,
52 rounds,147 accepted drafts and5 recomputations. Previous compaction was
10.906246956; this single-run difference is neutral, not a meaningful speedup.
No original numerical threshold or route changed. The candidate is the basis
for the separate source-tested staged-activation experiment, not a deployed
performance recommendation. At the user-requested stop, `PUBLIC_PROBE` is false;
original accepted compaction is selected. The frozen v196/v197 artifacts retain
their recorded experiment selections for future comparisons.
