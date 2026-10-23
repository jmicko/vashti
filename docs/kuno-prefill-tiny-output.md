# Coupled tiny-output producer and original CMMA consumer

Complete optional source continuation,2026-10-23; not built or GPU-tested yet.
Retain the ENTIRE GR-wide/expert/cohort/bank/GDN/QSA/frontier/prepared bundle.
No API/model/context/config/frontend/release change; Vashti:none.

## Measured lead and exact complete change

The preceding full GR-wide bundle passes257CPU/15runtime/429WGSL/304layouts/
353GPU/BOTH original chains/recovery but is speed-neutral. See
[kuno-prefill-gr-wide-tail.md](kuno-prefill-gr-wide-tail.md).
HC remains~0.90s versus reference HC185ms (instrumented scopes differ).

Actual original native product takes its masked K16 staging arm whenever N<16.
For the genuine BF1610240x4 HC injection this executes1280 workgroup barriers
per8-token tile. BF162560x1 shared gate (also retained FP16 variant) executes320 per8-token tile. This is a
source-based bottleneck lead, not measured isolated attribution. Strata uses
oneMKL; no assertion it uses this exact implementation.

strata-tiny-output / --tiny-output requires the complete preceding bundle.
TOGETHER: a real raw-U32 GPU producer materializes exactly16 weight rows, then
a new logical-output writer calls the SAME ORIGINAL chronologicalK16/8x16/F32
product with physicalN16. Full8-token tiles can use its original direct-global
arm; masked input M tails still use the unchanged original staging arm. Logical
output staysN4/N1 with original finite/zero/sticky fault and output mask.
No product/reduction/dtype/precision/participants change. No fake16-row source.

Reuse the existing real80MiB working-weight stage, disjoint from persistent GR,
outer, GDN/QSA/MoE views. No new production GPU or host allocation and no model
weight duplication/cache/budget change. BF16 source copies its exact original
4 rows including NaN payloads, then zeroes12 rows. FP16 uses the authentic
already-dequantized working prefix; an in-place raw producer preserves original
words and only zeroes15 extra rows. Disjoint source copy also supported for the
public control. Reject partial overlapping sources, wrong streams, malformed
bounds/offsets/strides and unsupported shapes. Source/workspace/consumer views
are REAL handles; normal dependency tracking observes producer-write then
consumer-read of the exact same327680/81920-byte view. No completion authority,
barrier bypass, fabricated scope, new disk cache or caller persistence.

Integrate only after the original packed-GR input and projection owner checks.
The general shared-gate path still executes its original rounded dequantization
and input selection before padding. Old general matrix paths remain intact.
New consumers count only fixed public diagnostic labels; private collector off.

## Required evidence

Retain all257 App controls plus3 actual tiny controls: independent source/zero/
publication lattices, true logical owners/alias/stride/hardware-input selection,
and real raw producer compilation with all16 layout variants/no matrix or SLM.
Retain15 runtime controls and429WGSL/304 matrix layouts; add16 real cooperative
consumer layouts (320 total), plus16 separate raw producer layouts validated by
spirv-val without weakening existing CMMA requirements.

Add40 actual paired original-GPU +independent original F64 controls to353:
BF16HC counts1/3/7/8/9/45/512/513 finite +NaN-weight/input at9 (copy mode);
BF16shared same10 cases in disjoint-copy mode; retained FP16shared same10 cases
in BOTH disjoint-copy and authentic-prefix-in-place modes.
Compare original/new bits for EVERY output/fault owner word, unchanged F64
1e-4+3e-4*abs(reference), logical output padding0, raw original source prefix,
all padded weight rows0 and full scratch tail sentinel unchanged. Preserve
original source/input images and source NaN payloads; no Session or false scope.
Then BOTH original18-script full model/profile/normal/compiler/owner/erasure/
physical-recovery chains before speed claims. All353 old controls stay.
No CPU compilation during GPU timing. Keep original clean-selection and staged
installation gates. No performance/install claim from source implementation.


Initial source1f28f1ab CPU build stops on Rust inner doc comments in an include!
file (E0753), before any test/GPU execution. Change those two comments to ordinary
comments; no algorithm, guard or test change. Preserve initial diagnostics.


Source7f3a94cd/workerdbdc14c6 passes260CPU/15runtime/429WGSL/320matrix plus16raw
producer layouts, with no GPU run yet. Final pinned profile inspection shows
DTYPES_10=[30;49] for current shared gates too. Extend the actual BF16 shared
selection and10 original-model GPU controls before testing, retain FP16 prefix
path. Original dtype-specific mixed input selection and source/dequantization
are unchanged. Planned all393GPU =original353+40 new paired controls. Counter
classification uses logical rows4/1 so BF16shared is not mislabeled injection.


## First complete tiny-source attempt

Sourcee7eee86b/workere92f1144 passes260CPU/15runtime/429WGSL/320matrix+16raw
producer layouts. Protected original operators reach the tiny suite; bitwise
original/new output and fault checks pass up to a full scratch-owner guard
failure. No full-model or speed run; physical recovery passes. Preserve
strata-parity-tiny-output-bf16-shared-20261023-operators and compiler/error log.

The new fixture uploaded padWords+64 sentinel words but compared EVERY remaining
physical owner word with the sentinel. BufferLayout::new rounds sub2MiB spans
to a power of two; Upload::words zeroes its remaining allocator padding. Correct
check keeps the64 uploaded sentinel words unchanged AND requires every further
owned word zero. No shader/math/private-path/owner/tolerance change. Add one CPU
control using the ACTUAL BufferLayout contract and actual extra padding; retain
all260 prior controls. New intended261CPU, unchanged15/429/320+16/all393GPU/BOTH.
Production allocation claims here mean inference payload owners; ordinary
per-launch runtime argument metadata still exists. No new payload allocation.
