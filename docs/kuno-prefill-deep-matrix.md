# Deeper chronological matrix stages

Backend-only optional `strata-deep-matrix` continuation,2026-10-23. Root owns
shared feature/CLI wiring, the actual client/compiler capability query,
integration, builds and hardware. No API, model, context, configuration,
frontend, installation or release change; required Vashti:none.

## Expert macro64 component

The original64x64 expert macro already stagesK64,notK32. The separately scoped
`native::deep_macro64` component stagesK128,keeping128 threads/eight16-lane
subgroups,the original four8x16 F32 accumulators,FP16 operands and exact
chronologicalK16 products. GU K2560 and DOWN K640 are divisible by128; no new
K-tail or arithmetic is introduced. Masked M tails,64..255-member admission,
original N dimensions,whole map/status/fault/source/output owners and routed
finite/zero/fault publication remain identical.

Both operand arrays have4096vec4<half> elements:32KiB each,64KiB simultaneous
SLM. The existing final uniform full-workgroup handoff barrier precedes
`a.free()`/`b.free()`; only THEN is the original16KiB publication storage
created. An explicit-workgroup-layout compiler has64KiB physical peak; the
nonaliasing layout remains80KiB logical storage. All eight compiler-layout
variants for BOTH roles are independently checked. These additional16 modules
remain separate from the unchanged original compiler/definition population.

Selection requires original geometry/alignment AND hardware>=64KiB AND
`client.native_explicit_shared_layout()`,which root obtains from the exact
actual server/compiler and conservatively caches per client. The preexisting
K64 macro remains the complete fallback. A64KiB hardware limit alone never
proves explicit layout. No public caller flag or guessed device model enables
aliasing. Original `expert_macro64_products` counts both implementations; new
public-only `expert_macro64_deep_products` counts actual new launches.

The new generated type has a genuine separate full name under
`native::deep_macro64::expert_macro64::ExpertMacro64`; the existing original
native grammar authenticates its unchanged expert_macro64 segment,role/grid,
INFO and six phases. Original actual kernel definitions remain untouched.
No shader is renamed to impersonate a different arithmetic family. Original
group-end markers,raw releases,copies,source/owner enrollment and erasure stay.

The new producer AND every new cooperative reader use K16-panel-major SLM,
not a deep linear128-scalar row pitch. Macro operands contain eight K16 panels
per bank; each panel has64x16 original half values. A job's physical vec4 index
is bank*2048+panel*256+row*4+within_panel/4. Cooperative loads use stride16.
The same original global vectors and chronological K16 fragments are read;
there is no padding,extra staging owner or arithmetic change. Independent CPU
mapping controls reconstruct original source addresses from logical panel
coordinates and check every real write/read owner,including masked M tails.

Four additional CPU controls cover real64KiB/explicit-layout geometry;
all actual vector owners/masked M tails for both banks and all original six
counts; identical original entry guards/output writer plus K16 order and
SLM lifetime boundary; and actual compiler products/loads/stores/barriers/
family geometry across16 layouts. The compiled modules require128 threads,
subgroup16,32 chronological products,32 cooperative loads,four original
cooperative publication stores,three original static barriers and sticky
fault operations. Additional offline validation canonicalizes and runs
spirv-val; marker `strata_deep_macro_shader_accepted` appears16 times.

The existing20 macro64 GPU controls retain their original F64/full-owner/
status/fault oracles and now additionally REQUIRE the authentic new path under
this feature,preventing a silent fallback from counting as deep-kernel proof.
No old tolerance or population is relaxed. Full model routing counts,
actual zero-spill ISA,both original complete chains and physical recovery
remain required. Source implementation is not GPU or performance acceptance.

## Routed128x128 prefix component

The separately scoped `native_expert_wide::deep` producer doubles its original
K32 stage toK64,retaining512 threads/32 sixteen-lane subgroups,the same four
F32 accumulators per subgroup,original GU/direct-GU/DOWN publication helpers
and original prefix/tail plan. Each stage contains four physical K16 panels;
each panel has128x16 original half values. Stage and register-prefetch writers
use bank*2048+panel*512+row*4+within_panel/4; cooperative readers use stride16.
Two genuine512-job rounds cover every operand vector exactly once. The eight
next-stage vec4 values stay in registers across four original K16 product
groups,then enter only the disjoint next bank before the unchanged handoff.

Two32KiB operand arrays and separate subsequent16KiB publication have the same
64KiB actual explicit/80KiB logical nonaliasing layout as the deep macro.
The final full-workgroup barrier precedes BOTH frees and scratch creation.
The runtime selects the new prefix only after ALL original source/map/shape/
CMMA/grid/view/tail-argument preflights and the same actual64KiB+explicit
compiler capability. Otherwise it queues the original prefix. The original
tail functions and real borrowed tail view remain unchanged. No fallible
conversion is postponed past either dispatch.

Original K32 functions,definitions and compiler controls remain available and
unchanged. New genuine types under
`native_expert_wide::deep::expert_wide_{gu,gu_direct,down}` require the root-owned
feature-gated EXACT closed-prefix extension to native grammar; original
role-bearing suffixes,grids,INFO and owner/phase checks remain mandatory.
The existing profiler recognizes their original exact role-bearing segments;
no profiler or metadata spoof is added. Public-only
`expert_wide_deep_products` counts actual new prefix launches. Old full-prefix
and tail counters,copy bytes,source roles and routing populations stay exact.

Four additional CPU controls cover original shape/workgroup guard composition
with actual capability; complete two-bank stage+prefetch+CMMA original-vector
ownership; original header/publication-helper identity,K16 chronology and SLM
lifetime; and all THREE genuine role entries across eight actual compiler
layouts. Each new module requires512 threads,subgroup16,16 ordered products,
16 cooperative loads,four original publication stores (eight compiled stores
for the direct/fallback branch),ten unchanged static barriers and sticky
fault operations. The shared actual CFG proof additionally requires eight
StorageBuffer loads before the genuine product block and eight Workgroup
stores after it,with its actual subsequent full-workgroup barrier. The
original8-product/4-load proof and its original48 accepted modules are retained.

Separate offline marker `strata_deep_expert_shader_accepted` appears24 times;
CFG marker `strata_deep_expert_register_stage_accepted` appears24 times. The
original compiler population is not replaced by these new modules. Existing
full-owner/F64/finite/fault/routing/tail GPU fixtures remain unchanged; the
wide fixture requires the actual new capability BEFORE queuing its prefix.
These source controls require root integration,CPU/offline checks,actual
zero-spill ISA and full GPU/model/recovery validation before acceptance.

## Performance expectation and limits

The macro stage loop has20/5 handoff barriers for GU/DOWN instead of40/10,
and the wide prefix40/10 instead of80/20,plus the same initial/publication
barriers. Total original K16 matrix products
and source bytes are unchanged. Fewer loop/barrier/address operations are an
optimization hypothesis; addedSLM/register pressure and native scheduling can
offset it. Pinned Strata's active Intel path calls oneMKL FP16xFP16->F32 in
`sycl/src/prefill/gemm.dp.cpp:799`; its internal tile sizes are not established
by this source. Do not claim this exact stage size is copied from oneMKL or
predict whole-prompt improvement from a single sampled layer.

## Dense K64 and complete operand panels

Root's dense component preserves all original K32/pitch40 definitions, public
controls and fallbacks. New direct M128xN128,K64 FP16/BF16 producers use512
actual SG16 lanes and exactly64KiB double-buffered A/B operands, without any
publication scratch. Both producer and matching cooperative readers use four
K16 panels of128x16; next-bank register staging uses that same physical map.
This avoids a raw64-half SLM row pitch without adding padding. Original K16
arithmetic/order, authentic vector views, actual output strides and the
unchanged separate finite/zero/sticky-fault consumer remain.

Selection requires genuine64KiB capacity, complete N128 tiles,K divisible64,
original CMMA/subgroup/thread/grid/owner checks and the original complete direct
store admission. The original GR320 projection keeps its accepted HC fusion.
The complete real M128 prefix is followed by the original genuine shifted tail,
without fabricated tokens, a larger source view, or an extra payload allocation.
Original explicit try_dense/try_dense_split entry points remain intact for their
original public/source controls; production selects the new path separately.
The actual dense_direct_store counts describe direct publication; old
`dense_slm40_products` is NOT incremented for the new representation. New
`dense_deep_k64_products` counts actual launches. Their sum conserves real
producer coverage, rather than claiming that the old padded code ran.

Five CPU controls cover source/stage/output coverage, independent K16-panel
reader coordinates, exact K16 chronology/final prefetch, real capacity/shape
rejections, and actual compiler modules. Sixteen additional native compiler
layouts retain64KiB physical size with/without explicit layout, genuine SG16,
16 chronological products,4 global cooperative stores,zero producer atomics
and actual prefetch/product/SLM/fence CFG. The original populations remain
unchanged. Twenty-one additional full-owner GPU cases compare actual deep
production and original producers/F64 oracles: FP16/BF16, output stride/offset,
finite/nonfinite,512/640 full calls,128/148/513 prefix/suffix calls including
poisoned suffix, and shortK64/128/320. Source-only until root acceptance.

The client-specific readonly shared-layout query reads the actual Wgpu server's
compiler options and caches only its Boolean in a clone-shared OnceLock. No
request/model bytes, external API or GPU authority are created. Unsupported
servers conservatively returnfalse. Expert deep variants require this actual
explicit-layout flag AND64KiB, because their later16KiB publication allocation
must reuse dead operands; dense direct publication needs no such aliasing.
