# Immutable public frozen GU stride-four integration

This source integration explicitly selects the scheduling candidate described
in [the leaf contract](kuno-native-frozen-gu-integer-stride-four.md). It changes
neither the declared integer arithmetic nor the original reference tolerances.
Physical CPU13/reference, real repair and matched full-language timings remain
required before accepting the candidate.

`Sources::gu_stride_four` is a private immutable boolean independent of the
original `Geometry`. It can be true only when `Sources::gu_integer` is true;
all existing integer policy conflicts still reject before source import. It
does not authorize ordinary CPU, live-member, prepared or private inference.
Every original source identity, partition, physical owner/stream/no-alias
proof, GPU router audit and completed family transaction remains required.

The existing `ResidentModel::attach_public_host_sources_with_gu_math(...)`,
`PreparedPlan::attach_with_gu_math(...)` and
`Sources::new_with_policy_and_gu_math(...)` signatures remain wrappers that
explicitly pass false for stride-four scheduling. New opt-in methods append
`gu_stride_four` immediately after `gu_integer`:

- `ResidentModel::attach_public_host_sources_with_gu_schedule(...,
  three_members, gu_integer, gu_stride_four, cancel)`.
- `PreparedPlan::attach_with_gu_schedule(..., three_members, gu_integer,
  gu_stride_four, cancel)`.
- `Sources::new_with_policy_and_gu_schedule(..., three_members, gu_integer,
  gu_stride_four)`.

Each selected layer prepares the original 64 KiB owner once, invokes the actual
`launch_group_stride4(...)` leaf on all five original GU source partitions,
retains each original DOWN call and finishes the full-owner integer audit.
Only each successful alternative leaf call increases a local stride-four
dispatch count. The sealed `GuIntegerDispatch` completion witness requires
exactly five such calls when selected and exactly zero when unselected. It
also retains exact layer 0..47, count 1..3 and all five integer dispatch checks.
The witness is accepted only through the original pending-router and full
model transaction. Selection alone and DOWN dispatches cannot credit it.

`Counters::gu_stride_four_dispatches` counts those accepted actual calls.
`ResidentModel::native_frozen_source_gu_stride_four(&executor)` first verifies
the healthy completed native projection boundary, then validates source
selection against positive integer preparations, audits equal preparations,
integer dispatches equal five times preparations and stride-four dispatches
equal integer dispatches. Unselected sources require zero stride-four credit.
There is no permissive geometry/math/scheduling fallback.

No new owner or precharge is needed. The integer target MAIN extent remains
828,932,096 B with the same full 64 KiB scratch, HOST increment 65,536 B, INFO
increment 196,608 B and runtime increment 462,848 B. Existing dynamic source
metadata bounds remain enforced. Default constructors and ordinary wrappers
select false, and unchanged warm lifecycle means explicitly selected sources
also use the candidate for the warm initial count 1 token.

Focused CPU tests additionally cover all source-selection combinations,
missing/extra/wrong-policy actual receipts, positive joined completion counters
and rejection of zero or mismatched credits. No controller, public API,
Vashti, installed service, GPU reset or release behavior changes in this
integration. Root owns explicit lab selector registration and physical tests.

## Source validation receipt

Source commits are `b751403` (separate scheduling leaf) and `42a0c62`
(immutable parent integration), based on `e10e9fb`. On 2026-10-11, all 13
focused integer GU tests passed: seven leaf tests and six parent/owner tests.
The command was `cargo test --manifest-path
apps/kuno/benchmarks/native-text/Cargo.toml --locked --features direct-device
--bin kuno-text-lab -j2 frozen_gu_integer -- --test-threads=1` using the shared
oracle target. Strict `cargo clippy` with the same manifest, locked dependencies,
direct-device feature, `--all-targets -j2 -- -D warnings` also passed. Formatting
and `git diff --check` passed. All compiler/test jobs were stopped by 15:55:58 UTC.

These receipts use the original per-quartet decoder from `e10e9fb`; they do not
include the independent encoded-register preparation candidate `6992246`.
That source composes through the shared arithmetic BODY but requires its own
integrated verification. No physical GPU correctness, original-reference
acceptance, full-language speed, installed service or release receipt is
claimed here.
