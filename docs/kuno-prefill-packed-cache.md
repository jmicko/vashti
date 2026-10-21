# Coupled packed-cache prompt attention

Backend-only optional candidate, not installed or speed-accepted. Retains the
complete copy-cohort/segment-compaction bundle and every earlier component.
No model/quant/config/API/KNT2/KNR2/Vashti change. Reference kernels unchanged.

## Evidence and exact scope

After copy cohorts, warm profiled4245 prompt9.7413s: full attention2.9552s,
attention/merge family1.5050s (essentially unchanged from1.5063s). Expert capture
CPU809.7->574.3ms; actual expert segments37959->10704 and reencoded commands
40107->11409. Avoid attributing remaining entire wall time to host enqueue.
Clean first12.7915/repeat9.2446s misses original5%-at-EACH installation gate;
keep correct copy-cohort component. Same-host Strata repeated reference5.3515s.

Select `strata-qsa-packed-cache` / builder `--qsa-packed-cache` only with original
complete compact/paired/uniform/FMA/gather/fault bundle. It changes only selected
PromptAttention and PromptAttentionFault sources BEFORE local fault aggregation:

- Read paired K words from one vec2<u32> in the SAME full cache descriptor.
  Each original264-word row and every original eight-value K offset is even;
  original byte extent, whole owning resource and hazards remain unchanged.
  Scalar V/scale extraction uses the corresponding lane of that same pair;
  the adjacent word stays inside the original immutable owned row.
- Use exact signed-byte bitfield extraction; same input bits and masked index.
- Check each shared selected FP16 K scale once instead of testing each of its
  eight signed-byte products. Every finite half scale times any signed byte is
  finite F32 (absolute value at most8,384,512). Any nonfinite half scale makes
  every product nonfinite, including0*Inf. This is the SAME sticky fault result,
  not removal of poison detection. All16 actual products, original chronological
  dot/FMA/reduction/softmax/PV math and subsequent finite checks remain exact.

No matrix/XMX path or register-router candidate is selected. Original scalar
control, decode, MTP, merge, causal selection, scores/stats/status, barriers,
resource ownership/flight/erasure and memory reservations stay intact.

## Acceptance

Before GPU execution require whole-source inverse outside exactly these changes,
all65536 half bit patterns times256 signed codes for fault equivalence, signed
byte extraction with adversarial neighbors, row bounds/alignment, actual Naga /
SPIR-V vector loads and signed extraction, identical geometry/bindings/barriers/
subgroup operations, complete selected source composition and original168 QK +12
PV FMAs. Keep all original166 selected App controls and11 unchanged runtime
controls,421 WGSL/216 matrix inventories and all189 original numerical GPU tests.
Add four separate actual-GPU cache-scale poison controls (positive/negative
Infinity, NaN, zero-times-Infinity), using the same original scalar shader and
EXACT original bitwise/full-owner/fault comparison. Keep the original twelve
attention fixtures and original129 operator count intact; report extras separately.
Original exact compiler/F64/status/fault/source/population/owner/budget/recovery
proofs and BOTH complete73/4245/4245 modes remain mandatory. No fresh install
unless original terminal/clean-at-EACH/staged/privacy/quality gates pass.

## Frozen CPU/offline checkpoint (2026-10-21 23:38UTC)

Sourcefe57978532b7842dd2df5e77aa7018207df6fac0; worker SHA256
`eaeb67cde05babda2cdc8ffd8e59cc9733272048dab9d74f2587ad5b01b9fc5e`,
build `strata-parity-packed-cache-20261021`. Final exact-feature170 App controls
PASS (original166 plus four new controls); original11 runtime controls rerun
PASS with byte-unchanged entire runtime source. Frozen421 WGSL/216 matrix
layouts PASS using original pinned2026.1 validator. No threshold/reference
relaxed. Extra four cache-poison GPU controls stay separate from original129,
plus original42 exact-role and18 direct-GU checks. Original full proof chain
and both profiles retained. Root owns `kuno-packed-cache-20261021.service`;
GPU correctness, actual compiler proof and whole-model speed pending.

Evidence in `/media/jmicko/faststore/kuno-development`:
`native-packed-cache-{cpu-selection,cpu-features,cpu-final,runtime-cpu,runtime-source-diff,offline,release-build}-20261021*`,
`check-native-packed-cache-connected-20261021.py` and
`native-packed-cache-connected-20261021.log`. The initial four-control diagnostic
also passes; exact final source includes the additional real GPU poison cases.
No install/API/frontend/config/release/reset or performance acceptance claim.
