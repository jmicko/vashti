# Unused fused four-row MTP cache repair

This internal leaf is based on `45c0b26efc865952b35c62ce6ff253088011aa7c`.
It was source-validated without a controller in `7ea302c` (root `bd46b2b`).
The approved separate controller is now implemented in `fc0948c` (root
`83967bc`); see [its contract](kuno-native-public-four-batched-mtp-repair-controller.md).
Protected activation and actual model GPU acceptance are still pending.
No installed selector or public API is changed.
`repair_accepted_cache` and `AcceptedCacheRepair::new` retain their original
one-to-three-input behavior, including rejection of a third-draft cursor.

The separate `repair_accepted_cache_fused_four` entry requires the actual
workspace's fused sweeps, original healthy full-Q8 source, no English shortlist,
no live draft output, and an authentic completed `LastForward` for this binding.
Its cursor must equal `target_start + 2`, proving that three real draft cells
have completed. It accepts only an actual accepted target prefix of one to four
tokens and exactly that many finite target carry rows. Every token, preceding
carry, row span and context position is validated before mutation. Prefix length
alone cannot establish a third draft. Tails with fewer drafts continue to use the
original API. The caller retains actual target acceptance and numerical authority.

Both wrappers share the existing queued rewind and scalar cache-append body.
After one whole-suffix rewind, each accepted row after the first uses the original
scratch clear, full guards, condition/HC/KV math, RoPE position and carry from the
preceding accepted target row. Fault stays sticky across internal clears. The
stored cursor, completed traffic and ledger publication wait for one final
full fault gate, source validation, cancellation check and completed fence.
Original error/panic cleanup wipes all cache owners and poisons the sequence.

The original wrapper keeps its actual 512-dispatch limit. The new wrapper checks
its actual accumulated dispatch counter against 222 before the final gate or
publication: six initial fused clears, one rewind, eight guards, and at most
three cells each containing seven clears, sixteen guards and 46 original math
dispatches. Three non-fused cells would require 590 and are explicitly rejected
before mutation. Neither wrapper resets dispatch accounting between cells.

All 42 owners, original cache geometry, 512-record/1 MiB INFO reservation,
16 MiB private and host caps, and 64 MiB runtime cap remain unchanged. Target
carry rows are borrowed, and the original one-row conversion/upload guard is
released on each loop iteration. No extra scratch, shader or mathematical
candidate is added. Existing output leases, source/physical proofs and closing
audits remain mandatory.

Required source checks cover prefixes 1/2/3/4, exact cell positions and carry
bits, old-API rejection, non-fused/stale/mixed-binding/cursor/span/finite/context
failures, actual dispatch accounting and sticky-fault/error cleanup ordering.
Original MTP repair/cache tests remain controls. This contract alone grants
no GPU acceptance. The separate controller composition has been approved and
source-tested; guard integration and unchanged protected CPU13 full-vector/
greedy/repair, closure/recovery and paired normal timing gates remain pending
at the user's paused checkpoint.

The v177 four-row record contains 155 drafts, 52 rounds, 147 accepted drafts and
five necessary target recomputations. A future joined repair could reduce 199
scalar fault reads to 52, saving 147 full 32 KiB readback/audit windows
(4,816,896 bytes). Draft packet reads and all target recomputations remain.
This is a count-based opportunity, not a measured performance improvement.

Source validation uses the existing `oracle-native-target-20261009` cache and
standalone native-text manifest, `--no-default-features --features direct-device`,
`-j2` and two test threads. All 47 resident-MTP tests pass: seven new fused-four
checks, eight unchanged joined-repair controls and 32 other original resident
checks. The nine original MTP kernel/math/binding tests also pass (56 total).
The original append/rewind control validates Checked/Unchecked WGSL and runtime
SPIR-V with `zero_initialize_workgroup_memory=None`. No device is constructed.
A source comparison confirms 16 original scalar/math/clear/guard/failure bodies
and the old repair preflight unchanged. The shared queue/catch/failure body is
identical apart from the actual dispatch-limit check immediately before its
original full fault gate. Strict pure-direct all-target Clippy with warnings
denied, scoped formatting and diff checks pass. Physical acceptance remains
pending; this commit does not select the leaf in any controller.
