# Fixed public target-only language controller

This benchmark answers how fast the original B70 target runs without waiting
for an RTX3060 MTP draft. It changes neither ordinary inference nor default
speculative lab execution. Missing `KUNO_LAB_LANGUAGE_DECODE` retains the
previous speculative path; canonical `target-only-public` selects the new
path. Empty, malformed, duplicate and inherited experiment flags reject.

The target-only selector is restricted to the existing compiled
`KUNO_LAB_LANGUAGE_PUBLIC_ONLY=1` `first-code` and `warm-prose` fixtures, fixed
8GiB static cache, direct-device-only build, and these original target flags:

- `KUNO_LAB_DEVICE_DISPATCH=inline`
- `KUNO_LAB_TARGET_FILL=direct-public`
- `KUNO_LAB_TARGET_DENSE=verify3-fused`
- `KUNO_LAB_TARGET_IO=chunk`
- `KUNO_LAB_TARGET_REUSE=request`
- `KUNO_LAB_TARGET_SOURCES=frozen-host-public`
- `KUNO_LAB_FROZEN_PARTITION_EXIT=1`
- `KUNO_LAB_TARGET_ROUTER=device-frozen-public`

Original chronological QSA selection and four-row DOWN may compose. Plain
original GU math is the default. The previously accepted bounded
`KUNO_LAB_FROZEN_GU_MATH=integer-public` and its explicit stride-four selector
may compose with the identical target prerequisites. This is a separate
target-only scope: it does not invent a fused draft-sweep selection. Normal
speculative integer math still requires the original self-contained Q8 draft
and fused sweeps. Three-GDN streaming is currently excluded because its
accepted geometry is three-target rows. Private/workgroup tables, shared DOWN
scales, GU factors, two-row GU, staging, three-member arithmetic, prepared
experts and live reductions are excluded initially.

Every `KUNO_LAB_MTP_PROFILE`, `KUNO_LAB_MTP_Q8_HEAD`,
`KUNO_LAB_MTP_VOCAB` and `KUNO_LAB_MTP_SWEEPS` presence rejects in target-only
mode, including selections that would normally mean a default. The original
`KUNO_TEST_MTP` path remains a required parsed input so the protected runner
can retain its transport shape. Target-only never opens or maps that file,
binds an MTP graph, enumerates/selects a Nvidia adapter, initializes a draft
runtime, loads draft weights or instantiates a draft model.

The real prompt uses the same original target chunks, weights and arithmetic
before source attachment. Full prompt logits and hidden packets are read; the
last real prompt row selects the first output by actual greedy decoding.
There is no teacher MTP KV. Each subsequent output executes the existing
target chunk routine with exactly one actual input token, reads full real
logits and hidden output, and checks binding, position, count and committed
target cursor. No oracle tokens, substitute routes or omitted model families
are used. Original compiled tokenizer/template, EOS behavior and the 200
output limit remain. A full run has one output selected during prefill and
199 timed target continuations. EOS can end the run earlier and is reported.

Timing retains first output emission, all real continuations, complete packet
reads, decoding, actual source/router checks and request closing audit. It
excludes the first output's prefill selection, model close and host source
detach, as the speculative fixture already does. Target-only has no draft
proposal, rejected prefix, state repair or draft erasure to charge. The normal
speculative path retains its existing verification and repair checks.

Receipts report `decode_mode=target-only-public`, `draft_executed=false`,
`draft_card=null`, `draft_profile=null`, `draft_public_reserved_bytes=0` and
`draft_close=null`. The record also reports absent `draft_traffic=null` and
`prefill_teacher_kv_executed=false`. Actual target-only counters require
`count1_chunks == completed_target_rows == emitted_tokens - 1`, at most 199,
positive full-packet timing when rows ran, and zero proposed/accepted drafts,
draft time and repair credit. Selected GU integer additionally requires
exactly 48 real preparations and owner audits per completed count1 chunk,
with five dispatches per preparation through the original validated receipt.

Full original target, all-family output/tail, SourceSet and immutable host
mirror ownership audits remain. Target-only final MAIN closes are exact:
1380 owners/828,866,560B with plain math, or 1381/828,932,096B with the existing
integer scratch. No allocation optimization is claimed in this comparison;
the original latest-target checkpoint reservation remains although no repair
transaction is executed. Source hashes and target file identity remain
mandatory through the protected runner. Source/content persistence is unchanged.

Parser, actual accounting and owner-extent tests cover canonical/default mode,
both public fixtures, every target prerequisite, missing/partial completions,
forbidden draft/experimental policies and strict close extents. Physical
target-only code/prose correctness, matched timings and Nvidia weight-traffic
absence are root-owned live acceptance gates. An independent CPU200 reference,
private requests and production acceptance remain explicitly false.
