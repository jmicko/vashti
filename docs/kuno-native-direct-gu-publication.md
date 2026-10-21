# Coupled GU publication and activation checks

Implementation in progress; **not yet validated or selected**. Keep the complete
matrix-attention and subgroup-publication work. Do not remove finite checks to
make cooperative stores appear faster.

## Evidence and scope

Subgroup publication source3228cd9e/SHA0e45b16a passes145 selected CPU controls,
420 WGSL/208 matrix layouts, original171 GPU numerical controls, both complete
original profiles and all compiler/count/pool/source/owner/recovery proofs.
Normal first/repeat11.645375742/10.189380043s versus the same scalar/pool
predecessor11.608595581/10.128802681s is not a measurable gain. It stays
unselected; no new clean, staged or installation acceptance was run.

The native M128xN128 kernels currently store every cooperative fragment to a
subgroup's128-F32 SLM slot, synchronize, read it back, check/sanitize each value
and publish it. This remains an actual round trip after narrowing the barriers.
GU output is contiguous at member*1280+feature. DOWN output instead scatters
through the original route map and retains its status policy. Target only GU
wide prefixes first; keep DOWN and checked standalone primitives intact.

The next activation already reads every GU element. A direct cooperative GU
store must be implemented together with its mandatory validating activation,
not tried separately. Eliminating scratch might reduce GU shared storage from
49152 to32768B. Actual compiled envelopes must prove that; no hardware occupancy
or speed gain follows from this calculation alone. Do not sum sampled GPU class
intervals as isolated kernel costs or attribute all end-to-end time to them.

## Exact semantic boundary

The original checked GU primitive sets fault128 and stores positive zero for
each nonfinite result. It remains available with exactly that immediate
contract for standalone callers and original reference controls.

Introduce a distinct internal direct-GU kernel only inside a genuine closed
native expert batch. It keeps original map/header, shape, device, whole-owner,
stream, grid, prefix/tail, INFO and original source checks. It writes its full
128-member prefix through cooperative global stores to the existing GU owner;
it does not expose an independently valid unchecked primitive.

Its mandatory activation must, before computing SwiGLU:

1. Check all four original GU values for each existing two-output invocation.
2. Replace each nonfinite GU value with positive zero in the original GU owner
   and set the original fault128. Preserve finite values, including signed zero.
3. Run the original activation, infinity-saturation, F16 packing and status
   expressions on these sanitized inputs, with unchanged operation order.

Keep the original activation `good`/status policy after sanitization. A bad GU
input already sets fault128, but must not independently change `status[route]`
relative to checked GU followed by original activation. In particular, do not
replace the original `hf_sat` behavior with a generic all-finite test: activation
infinities saturate; NaNs remain faults. Positive and negative zero also matter.

After the coupled pair completes, the complete GU bytes, packed hidden output,
status, fault and original padding must match checked GU + original activation.
The checker covers the full active group, including any existing checked tail.
Do not add a GPU pass; fuse the check into the activation that already consumes
these values. Private failure/erasure behavior stays original even if failure
occurs between the two stages.

## Authority and complete pipeline changes

Use distinct role-bearing kernel identities. The closed original six-phase
collector must require the validating activation after a direct prefix and its
required adjacent original tail. It must reject plain activation, missing or
reordered checking, mismatched GU/fault/map views, incomplete pairs and unrelated
commands before GPU submission. A flag or fabricated public/prompt scope is not
a substitute for the original active batch/owner/epoch authority.

Checked standalone matrix calls and original numerical references keep their
immediate sanitization behavior. Direct publication may not silently replace
those calls just because a build feature is enabled. If the genuine coupled
context cannot be established, select the original checked implementation.

The new activation genuinely writes GU when sanitizing, so its binding, actual
SPIR-V write inventory, sealed access manifest, graph declarations and collector
view validation must all reflect that read/write access. Never report a read-only
buffer to avoid a hazard. Preserve complete original owner ranges, allocator
retention, phase dependencies, queues, real completion and quarantine. No
scheduler-only narrowing, cached private certificate, invented completion or
weakened source mutation guard.

Retain original8MiB command tape,4096 views,32 exact pipelines,24MiB parent batch
reservation and64MiB graph cap. Derive actual simultaneous inventories; do not
increase limits to accommodate the experiment. Prefer replacing the selected
GU/activation roles rather than retaining both as extra active batch pipelines.
Logical phases remain six; report actual direct prefixes and validating
activations from actual accepted commands only in the existing fixed-public sink.

Both ordinary and retained-native compilation must preserve the same matrix
precision, actual global cooperative stores and original subgroup memory fences
where scratch is still used. Keep input-stage workgroup barriers. A new shared
envelope is proved per real definition/layout, not inferred by subtracting an
unused source variable from a constant. No new model, runtime dependency, network
API, KNT2/KNR2, configuration or Vashti consumer change is required.

## Acceptance

Add CPU controls for the genuine collector pair and all mismatch/reorder/escape/
epoch/view/capacity negatives. Prove original prefix/tail output bounds and
complete coverage, actual global cooperative stores, correct GU write access,
unchanged arithmetic/precision and all real compiled memory layouts.

Add genuine fixed-public GPU pair controls comparing the checked original pair
with the direct+validating pair, including full and nonzero-tail groups and
bounded NaN/+Inf/-Inf inputs on both gate/up channels. Use authentic private
owners and exact raw-bit uploads when finite-only convenience upload rejects
deliberate poison. Compare full GU, hidden, status, fault and owning padding;
close and verify complete owned erasure. A diagnostic fixture may not manufacture
production request/batch authority. Keep all original171 controls and their
existing bounds, and require real whole-model counters to prove the new pair
actually ran under the production closed-batch collector.

Then complete both original profile/proof modes and physical recovery. Only a
promising complete result proceeds to original terminal-negative, fresh clean
45/4245/4245 at-EACH5%, staged thinking/tools/MAX8/semantic/privacy/sustained and
official reversible installation. Preserve all failures. Do not change warmup,
exclude cold compilation, select repeat alone or replace original scalar/F64
oracles. Installed worker3fffc266/SHAeac78ce0 stays until acceptance.

## Route-preservation implementation correction (2026-10-21)

Code review found that a raw cooperative global store would overwrite GU for an
invalid mapped route. The original checked kernel leaves that member's GU bytes
untouched and sets fault128. The coupled activation cannot restore overwritten
bytes, so direct publication now checks the eight fragment routes with a genuine
subgroup vote. All lanes take a uniform branch: valid fragments store directly;
otherwise the unchanged checked publication handles the fragment, preserving
both valid rows and untouched invalid rows. Original shared scratch therefore
remains49152B. No reduced-memory/occupancy claim is made. The objective remains
eliminating the publication round trip on valid fragments, with unchanged guards.
Both real compiler paths still apply original subgroup shared-memory semantics.

## Coherent CPU implementation checkpoint (2026-10-21 00:31UTC)

`strata-native-direct-gu` / builder `--direct-gu` requires the complete native32,
subgroup-publication, scoped bulk and scratch-ring feature bundle. Production
selection queries the genuine scoped client; a closed/foreign/failed collector
rejects rather than executing immediately. The original standalone GU remains
checked. The closed six-phase grammar requires exact generated direct GU role,
its original adjacent tail and `StrataPromptExpertF16SwigluCheckGu` with original
map/GU/input/status/fault views, metadata and member grid. Actual accepted-command
receipts report direct prefixes and validating activations through the existing
fixed-public-only profile sink. No private content counters or persistence.

147 selected CPU controls pass with the exact release feature list, preserving
all145 subgroup controls and adding real direct-module/global-store provenance
and activation source/usage controls. All8 layouts retain49152B, original input
and subgroup fallback fences,8 matrix products,4 global GU cooperative stores
plus4 original scratch fallback stores. Original full/tail numerical controls
are unchanged. Naga's actual activation use inventory is READ|WRITE for GU;
retained graph registration derives its exact hazard access from that inventory.
The matrix's actual compiled output visibility is also ReadWrite. Original math,
hf_sat/status tail is byte-identical after the fused per-component sanitization.
11 controls over actual runtime grammar/capture sources pass, including sealing
on success/error/panic and paired role/view/INFO/grid/order/tail negatives.
Artifacts `native-direct-gu-{complete-cpu,cpu-selection,cpu-features,runtime-cpu}-20261021*`
live under the existing artifact base; all are CPU/compiler evidence only.

New fixed-public GPU controls are prepared, NOT yet executed. They compare full
GU/hidden/status/fault/source owners and padding for256/257 members across finite,
both gate/up NaN/+Inf/-Inf and invalid first/last routes, then reset/verify owned
erasure. They use the existing explicit public-audit arena; they do not fabricate
production prompt or batch authority. Production whole-model accepted-command
counters are separately required. Offline release, original171 GPU controls,
new18 pair cases, BOTH full profiles/recovery and original clean/staged/install
gates remain pending. No speed result, reduced SLM/occupancy or install claim.


### Preserved first GPU diagnostic setup failure

Frozen2babfde3/SHA6fe832e2 passes421 WGSL/216 matrix layouts. The protected
operator run exits1 in the NEW pair harness: the activation helper called the
production dispatch wrapper, which requires a genuine parent STAGES scope.
The standalone public-audit pair has no such scope; `invalid buffer length`
therefore correctly stopped it before any timing. The run recovered physically
and resumed admission; no reset/installation occurred. Evidence
`native-direct-gu-connected-20261021.log` and
`native-direct-gu-public-failure-20261021.json`. This does not prove the new pair's
numerical acceptance. The diagnostic helper now validates exact actual public
owner sizes/grid and launches its original activation directly, like the other
standalone controls, without inventing a production prompt scope. Production
parent dispatch and the real closed-batch selection are unchanged. Rebuild and
repeat numerical acceptance before profiling.
