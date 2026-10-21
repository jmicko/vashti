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
189 controls and four packed-cache poison cases remain. Actual full public
profiles must show increased native prefix dispatches exactly matching new
128..255 group counts; no path credit from CPU selection or invented receipts.

No compilation or GPU call until the frozen packed-cache full profiles finish,
to avoid contaminating that timing. No install unless original terminal/clean
5%-at-EACH/staged privacy/quality and physical recovery gates pass.
