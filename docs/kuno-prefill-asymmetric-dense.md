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
