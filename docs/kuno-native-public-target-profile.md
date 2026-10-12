# First warm target verification GPU diagnostic

`KUNO_LAB_TARGET_PROFILE=first-warm-chunk` is a default-off, fixed-public CPU13
diagnostic. It requires warm replay `once`, Intel target plus independent Intel
full-Q8 fused MTP, original frozen HOST Sources, actual GPU routing, partition
exit, direct fill, inline dispatch, fused Verify3, chunk I/O/request reuse, both
integer GU and integer DOWN, grouped PLE and original bank fill. Head absent or
`integer-public`, original three rows or genuine four rows, and their existing
scalar/batched repair compositions are admitted. All their existing exclusions
remain; unknown/empty/duplicate selectors, other draft/device/math/placement
scopes reject before model construction. No language or normal service selector
is added.

Root registration uses the public `Inputs.profile_first_warm_chunk` flag to
configure only Basic stdout profiling before target client creation, require
actual device timestamp features and `TimingMethod::Device`, pause that client,
and keep the global configuration set exactly once. The runtime intentionally
rejects a second configuration set. The independent MTP client inherits the
Basic logger but is immediately paused after its creation, before any draft
kernel, and remains paused throughout the run. Only the target client is armed
by the scoped guard. The probe also requires the actual Device method and pauses
target profiling before model construction. Default execution is unchanged.

Only the first real warm `LatestTargetRound::verify` is profiled: replay one,
zero previously completed target rounds. A scoped guard unpauses the already
configured target logger immediately before verify and pauses immediately after,
including unwinding/early errors. Original cold and warm first-token prefill,
drafts, public full-vector checks, rejected-prefix recomputation, repair and
later rounds are excluded. A diagnostic-only target sync after pausing emits
the Basic kernel-name/timestamp summary. No payload, route, prompt, token or
carry is added to logs. Existing numerical, finite, source and owner authorities
remain unchanged.

Each completed pass records `target_profile` with selected scope, actual timing
method and completed profiled-target-round count: cold/default false and zero,
selected warm true and exactly one. The outer requested selector is separate.
Selected profiling runs are explicitly not comparable normal throughput runs.
Basic profiling introduces diagnostic timestamp/query/readback overhead outside
the existing private arena. The pinned profiler resolves each completed span
through two 8-byte timestamp resolve buffers and one 16-byte readback buffer
(32 logical bytes), plus pooled two-slot timestamp query sets and driver/CPU
metadata. These are diagnostic runtime allocations, not model owners or
measured physical allocation sizes. It adds no admission credit;
all original caps and cleanup remain. Kernel duration sums measure this one
diagnostic batch, not exclusive whole-generation phase time or a speedup.

Source checks cover exact default/canonical scope and three/four/head/repair
composition, missing gates and alternate policies, first-round selection and
actual Device/count receipts, and guard pause on successful/failed verification.
Physical timestamp availability, Basic output, source/full-vector acceptance,
cleanup and recovery remain the root owner's gates. No GPU or service change is
performed by this controller task.

All four new focused checks and all 48 existing speculative controller checks
pass (52 distinct). Strict pure `direct-device` all-target Clippy with warnings
denied and two compilation workers, scoped formatting and diff checks pass.
The isolated source base is `e880ab5`; root registration/protected scope and
physical GPU evidence remain separate. No hardware timing was run here.

Protected runner validation also requires the same canonical integer-head
prerequisites, warm CPU13 marker and `once` replay; it rejects language markers,
other modes and non-MTP selection. Profiling is removed from the policy map
only after validation in the existing MTP branch. The original 48 GiB process
cap and all ordinary admission scopes remain unchanged. Four new focused Python
checks and all 73 existing protected-scope checks pass (77 distinct), with diff
checks clean. Root still owns main registration and the physical runner.
