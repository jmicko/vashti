# Asymmetric dense matrix candidate

Backend-only complete M256/N128/K32 candidate. No inference API, configuration,
model payload owner, precision, persistence or Vashti change. Source composition
and compiler/numerical/recovery acceptance must precede speed selection.

The new protected fixed-public Strata kernel profile confirms SIMD16 GEMMs with
512-thread groups, including several busy variants dispatched with32 groups.
It does not expose their register allocation or prove their internal tile loops.
Do not diagnose thread count alone or claim our proposed tile is oneMKL's tile.
Original source/benchmark target stays fixed; profile instrumentation is not a
clean timing reference. The first profile lost teardown output because Strata
uses _Exit; a fixture-only Rust shim flushes only the pinned profiler destructor.
It never enters Kuno, changes Strata math or invokes Strata/oneAPI destructors.

The existing Kuno M128/N128/K64 double buffer requires65536B and is unavailable
under this device's49152B Vulkan workgroup cap. Add a complete asymmetric tile:
M256/N128/K32,WG512,SG16.32 subgroups cover4 M8 fragments and2 N16 fragments each.
Two K16 slices per bank are physically contiguous, with genuine stride16 matrix
loads. A needs4096vec4 across2 banks; B needs2048vec4: exactly49152B. No output
scratch/alias assumption. Eight F32 accumulators retain original chronological
K16 products. Six next-stage vec4 values per lane stay in registers across the
current products, then publish to the disjoint next bank and join the original
whole-workgroup barrier. Last K32 iteration never reads beyond the input.

Producer mapping:512 lanes, two rounds cover A's256 rows; only first round
covers B's128 rows. One job owns8 original half values; no uninitialized cell
or padding read. Genuine BF16 GR320 B tail stays zero-filled, and its nonlogical
output fragments never store or set faults. Consumer uses M32/N32 per subgroup,
so paired subgroup coordinates cover M256/N128 exactly once. Direct global F32
cooperative stores feed the unchanged finite consumer/fault writer.

Host selects only aligned complete M256 batches, original admitted N128 shapes
(or existing authentic BF16 GR320), K32, WG512/SG16 hardware and exact49152B
capacity, original vector views/bounds/nonalias/grid checks. All unmatched forms
retain existing paths. Old K64 deep and every old compiler/GPU control remain.
Add real compiler layout/stride/CFG/binding/output proofs, independent staging/
coverage/guard CPU controls and paired original-shared/F64 GPU cases including
nonzero offsets/strides, finite and poison inputs, genuine GR320 and full-owner
holes/source/fault checks. Full-model routing/ownership/capacity/reuse/recovery
must pass before speed. New anonymous counter partitions dense producer counts
without changing existing direct-store/finite-consumer logical laws.

No source build, GPU acceptance, speed improvement or release claim yet.

## Composed source and precise acceptance populations

The complete shader/host/compiler/paired-oracle source is composed. Three new
CPU cases cover every A/B bank address, register prefetch destinations, original
K16 chronology/final-block bounds, full output coverage and capability negatives,
then all24 actual layouts for FP16, BF16 and genuine GR320. Compiler proof needs
49152 real bytes, WG512/SG16,16 products,12 shared stride16 loads,eight global
stores, two WG barriers and in-loop load/product/store dataflow in each layout.

Fifteen additional paired GPU cases retain all732 earlier controls: FP16/BF16,
M256/M512,N128/N256,K64/K128/K320/K10240, nonzero weight/output offsets and
strides, NaN/Inf input/weights and genuine GR320. Compare original matrix output
bitwise, independent F64 tolerance and full source/output holes/fault owners.
M256 uses the original macro/shared fallback reference; M512 uses original full
shared wide reference. No original test is replaced. Planned totals360 AppCPU,
18/49 runtime,8+9 metadata,24 new compiler layouts,747 GPU controls.

Old K64 availability remains false at48KiB; new capability/report is distinct.
Unsupported hardware reports unavailable rather than treating zero cases as
success. Full-model anonymous dense counts partition as
slm40 + deep_k64 + asymmetric = direct_store = finite_consumers.
No new feature flag; existing deep-matrix flag composes this complete path.
No changes to RAW/cohort/packed-source/attention contracts or retained capacity.

Reference profiling artifacts are under development:
`strata-kernel-profile-public-receipt-20261023.json`,
`strata-kernel-profile-public-gemm-20261023.json`,
`strata-kernel-profile-exit-20261023/trace/kernel-profile.1` and the pinned
`strata-kernel-profile-exit-harness-20261023/`. The profiler is a verified local
Conda-forge artifact, mounted read-only into the existing reference container;
no system install. Package2.4 requires a different XPTI ABI; compatible2.3 used.
First successful three-request reference run lacked profiler teardown output
and is explicitly not capture acceptance. The second flushes only the pinned
tool's exported destructor at Strata's normal _Exit and captures22 GEMM variants.
Original public fixture SHA/token counts and inference binary remain unchanged;
all three reference requests and physical recovery pass. Instrumented8.39s is
not the clean5.3515s baseline and must not replace it. Kernel properties absent
from this older tool (e.g. register-file size) remain unproven.

## First complete acceptance and register-pressure follow-up

Source d6c98298e32e309ede91b4fbb6da3e1314e777b9, worker
057f899fc49f279406f628edf2e6be16ed55a222787ecd797ce00b35ac72dfb6
passes360 AppCPU,18/49 runtime,8+9 metadata, all original compiler populations
plus24 asymmetric layouts/CFG proofs, all747 GPU controls, both complete-model
chains and physical recovery. All15 new pairs have zero bit/F64 mismatches.
Actual large-request partition475 asymmetric+475 old slm40=950 direct stores=
950 finite consumers; no K64 deep selection on48KiB hardware. Original732 GPU
controls and357 AppCPU cases remain exact prefixes/retained populations.

The first CPU attempt used the original symmetric checker, which derived eight
next-stage transfers from16 products. New asymmetric geometry actually needs
six. Preserve that rejected attempt under development
`asymmetric-dense-six-vector-check-failure-20261023/`. New explicit asymmetric
checker requires exactly6 actual StorageBuffer loads before products and6
Workgroup stores after them. Original symmetric minimum8 is unchanged.

Diagnostic first/repeated11.878727028/9.867797979s; normal schedule
12.032958780/9.788971544s. These are slower than preceding K32 normal
11.435514301/9.306249103s and the retained FP16-image control normal repeated
8.854793s. No CLEAN run, no parity claim. Latest CLEAN stays12.471177197/
8.394039699s versus Strata5.5006/5.3515s. Do not replace the target or conceal
these regressions. Complete source/worker/evidence remains available.

Actual asymmetric Intel programs spill heavily at128 GRFs: FP16/BF16
109:106 spills:fills, GR320122:132. Old8 wide programs remain zero-spill and
both QSA ISA hashes unchanged. Measured dense coarse regions regress together:
GDN projections0.403602→0.588714s, output0.162664→0.249223s; QSA projections
0.122782→0.173578s, output0.054956→0.084207s. Expert6.069373→6.068521s and
QSA attention merge remain essentially unchanged. These include queue waits;
not an isolated-kernel causality claim. Register pressure is the measured next
implementation target, not evidence that the complete geometry is invalid.

Next complete source shortens A-operand lifetime: keep both B fragments, load
one A fragment immediately before its two products, then reuse that one A
matrix for each of the other three rows. Original sixteen chronological
products, eight C accumulators, six-vector register prefetch, both shared banks,
all bindings/owners/guards, direct stores and finite consumer remain. Twelve
actual cooperative loads stay required. This follow-up is composed but not yet
built/GPU accepted; don't infer a speed gain from its source order alone.

Evidence uses development `asymmetric-dense-*` including build receipt,
acceptance freeze, both safe phase exports, complete ISA export and
`strata-parity-asymmetric-dense-20261023-expert-cohorts-public-summary.json`.
Protected unit invocation ada5de2dbede455a9ba1a5ef7e513668 exited0 after
12m41.884s, zero swap, peak34.7GiB. Independent21:57:38UTC state: Intel
50,688,000B,NVIDIA compute0, all5 classified faults0, HTTP200, admission normal.
No installed binary/API/config/Vashti changes. Goal ACTIVE/UNMET.
