# Complete native32 medium expert candidate

Backend-only optional successor to packed-cache and copy cohorts. No installed,
GPU correctness or speed claim yet; no API/config/model/Vashti change.

## Evidence and complete change

Copy-cohort sampled layer: original390 groups, matrix30.86ms, dequant19.70ms,
ordered copies29.04ms. Whole long prompt still performs27.31GB host transfers.
Wide native32 only covers groups with at least256 routed members:4688 matrix
prefixes across26673 groups. Groups128..255 fall back to original SIMD16 macro
products even though their complete128-member native32 tile fits existing views.

Extend the existing native32 exact128-member prefix + original checked tail to
128..255-member groups. Change ALL coupled selectors: host plan, actual shader
header, backend six-phase grammar, original source/owner and exact piece/grid
checks. This is broader supported tile geometry, NOT looser math or ownership:
all original numerical bounds, full member/image/status/fault views, byte caps,
route checks, exact shader identities, scratch49152B, real completion and erasure
remain. No additional allocation. Keep the complete previous candidate.

Original256-minimum policy stays separately testable with explicit minimum256;
actual selected medium policy uses128. Original CPU fixtures still exercise the
old policy, while a new actual-policy test covers every new count, complete
prefix/tail ownership, wrong minimum and shared-memory limits. Add real backend
grammar controls for exact128 prefix plus every original8/16/32-member tail,
reject incomplete/foreign geometry. Original11 controls remain unchanged.

Add22 separate actual GPU routed-product controls (GU and DOWN): counts128,129,
159,160,191,192,255; plus129-member nonfinite first/other and invalid prefix/tail
routes. Use the original independent expected IEEE products/tolerance, complete
output/status/fault/source/tail owner scans and authentic route builder. Original
189 controls and four packed-cache poison cases remain. Also add18 actual
Direct-GU +validating-activation exact-pair controls for128/129 members, using
all original nine finite/nonfinite/invalid-route variants, exact bits and full
GU/hidden/status/fault/source/padding scans and owned erasure. Preserve original
18 cases for256/257 unchanged. Total original+new GPU inventory233. Actual full public
profiles must show increased native prefix dispatches exactly matching new
128..255 group counts; no path credit from CPU selection or invented receipts.

No compilation or GPU call until the frozen packed-cache full profiles finish,
to avoid contaminating that timing. No install unless original terminal/clean
5%-at-EACH/staged privacy/quality and physical recovery gates pass.


Initial sourceffa332f8 passes171 selected App controls (all original170 +actual
medium policy),13 runtime controls (all original11 +two medium grammar tests).
Final source adds ONLY the corresponding exact Direct-GU paired GPU controls;
repeat exact-source CPU/release/offline before the full GPU chain. No new speed,
installation or source/ownership/numerical relaxation claim.


## Frozen final CPU/offline checkpoint (2026-10-21 23:54UTC)

Final sourceda5aae3627905fa858fc96e0200f95e187d498a2 / worker SHA256
`61f89376ff5273bd88ad697b9ef3ae7658db8fb0538b2231605ac8727faebe6d`,
build `strata-parity-medium-experts-20261021`. Exact selected171 App controls
PASS again. Runtime13 PASS; entire runtime tree unchanged from the initialffa
CPU proof, verified by empty diff. Frozen421 WGSL/216 matrix layouts PASS with
original pinned2026.1 validator. No reference/tolerance/cap/owner guard weakened.

Protected `kuno-medium-experts-20261021.service` now runs all original189 controls,
four cache-poison,22 medium routed-product and18 medium exact Direct-GU paired
controls, followed by BOTH full original profile/proof/recovery chains. New
public aggregate `expert_medium_groups` is collected only in existing fixed-public
profiles; ordinary private workers retain no counter/history. Acceptance demands
actual wide-prefix delta equals2*medium groups and direct-GU delta equals medium
groups against the exact preceding packed-cache receipt. No invented path credit.

Artifacts under `/media/jmicko/faststore/kuno-development`:
`native-medium-experts-{cpu-selection,cpu-features,cpu-final,runtime-cpu,offline,release-build}-20261021*`,
`native-medium-final-runtime-unchanged-20261021.log`,
`check-native-medium-experts-connected-20261021.py`,
`native-medium-experts-connected-20261021.log`. Original terminal/clean-at-EACH/
staged/install gates remain required. GPU numerical/performance results pending.


## First protected attempt: fixture allowlist rejected before new kernel

Sourceda5aae36 worker exited1 with physical recovery PASS before invoking the
first medium product: shared fixed-public fixture() only allowed its original
nine counts. No matrix numerical or speed failure occurred. Preserve the failed
receipt and compiler code (`native-medium-experts-connected-20261021.log`).
Extend ONLY the compiled-public fixture allowlist to the seven explicit new
counts when the feature is selected; keep original nine fixtures unchanged and
all other counts rejected. New CPU control validates the real original route map,
high sorted routes and finite owned images. Add static public error diagnostics
so a fixture error cannot be obscured by a generic stage label. Engine arithmetic,
selectors, header, grammar and all numerical thresholds remain unchanged.
Repeat final172 CPU/offline/frozen233-GPU/full-profile proof on a new artifact;
never reuse this incomplete attempt as an acceptance or performance receipt.
