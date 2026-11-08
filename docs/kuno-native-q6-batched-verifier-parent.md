# Batched Q6 integer verification head parent

This is an explicit default-off Intel fixed-public target candidate based on
`08aa577`. It changes the original F32 head arithmetic to the separately tested
two-residual-Q8 packed-integer arithmetic. Source acceptance is not GPU parity
or a throughput claim. Original cold CPU13, real prompt prefill, model weights,
full-vector tolerances, real speculative repair and all owning closes stay
authoritative. The old count1 `q6_integer_head` API remains unchanged.

## Selection and completed authority

`GpuExecutor::activate_public_q6_integer_verify_head()` selects the new distinct
`q6_integer_verify_head` policy exactly once after the model's original completed
prefill or accepted warm boundary and authenticated frozen HOST attachment.
The narrow ResidentModel activation/query wrappers require the unchanged completed native boundary. The controller owns the public fixed selector.
The executor checks actual active admission, DAS144 source identity, native MAX8
tile8 target rows, original Verify3/fused policy and original checkpoint/runtime
configuration. Selected count4 additionally requires the actual four-row
reservation and activation; member capacity remains MAX8, not4.

`public_q6_integer_verify_head()` returns selection and actual successful
prepare/tile/member counters. These are enqueue counters, not independently
completed model authority. A controller can publish them only after its healthy
whole-model boundary and exact joined request work: one prepare and16 tiles per
successful head, with actual active members accumulated after the final tile.
Selection alone grants no execution credit. Failed/partial launches cannot
produce a completed receipt. Counters remain monotone across ordinary rejected
prefix restoration. Terminal executor close resets policy/counters.

## Sealed scratch and execution

The new `BatchedFinalHeadScratch` capability is constructed only by the actual
prepared FinalAllRows/FinalFourRows GR consumer after original MIXED production
and status folds, when NORM has no later original reader. It carries the actual
arena, four distinct original IDs, active count and reserved target capacity.
It is neither clonable nor caller-constructible. The default trait method
ignores the capability and enqueues the unchanged original head.

The selected projector consumes the capability, checks actual full arena
layouts, same client/stream, physical full buffer extents and nonaliasing for
MIXED, NORM, logits and fault. It calls the new explicit batch leaf with original
`output.weight`, raw type14, H2560, V248320 and one expert. Source identity,
original file integrity/stat checks and cache keys remain unchanged.

One prepare quantizes every active row into NORM. The same original16 cached
tiles (fifteen16384 rows and final2560 rows) decode each weight block once for
all RHS. Input stride2560; planes stride1440 words; output stride248320.
Counts1..3 use the separate runtime-count batch entry; genuine4 uses its true
four-RHS entry. No repeated count1 launches, new weight mirror, replacement
expert or silent geometry/math fallback is allowed. Every original row is
covered exactly once and each member preserves its own complete output.

After all16 consumer calls, GR queues a clear of the **entire** original NORM
owner on the same stream. The original output packet, full fault/finite/gap/
padding proofs, family acceptance, source lifetime and complete closing erase
remain unchanged. Any intermediate failure keeps GR poisoned and uses the
existing full-model failure/close path; no partial healthy-reuse claim follows.

## Exact unchanged budgets

H2560 needs1440 u32 words per prepared member,5760 words /23040 bytes at4.
The original GR NORM owns81920 logical words /524288 physical bytes. MIXED is
MAX8 H2560; logits retains the actual original3/selected4 logical reservation
inside the same4MiB physical owner; fault is its original full private page.
No new MAIN/HOST owner or larger cap is introduced.

One32-byte prepare INFO and16 bounded48-byte project INFO records plus one
complete scratch clear need at most18 original2048-byte INFO envelopes. The
existing head reservation remains33 envelopes (`2*16+1`), including the full
source/target/inspection accounting already owned by runtime. Preparation is
admitted inside that fixed bound before private fill; no metadata-only quote
credits allocation or bypasses actual runtime I/O limits.

## Required checks and handoff

Parent source checks cover default-false/immutable selection, actual3/4 policy
versus MAX8 owner capacity, all16 original cache tiles, full four-owner roles,
planes fit and unchanged INFO budgets, inactive output spans and failure/counter
joins. Leaf independently validates packed math, original C fixtures, all real
counts, checked/unchecked shaders and finite/sticky/full owner complements.
Then root runs physical operators, unchanged original cold/selected warm full
CPU13 vectors/carries/greedy picks/real repairs and complete cleanup/recovery
before same-artifact compiler-idle normal timing. No target-head share or30
tokens/s result is inferred from host enqueue timings.

Parent implementation is source-ready on `08aa577` plus leaf `5a84c26`.
The new path is default false and does not activate itself. Seven new focused
parent tests plus fourteen relevant original GR/projection/scalar-head tests
pass (21 unique tests). These cover real count/constructor capacity, rejected
artifact/views and partial receipts, exact tile coverage, full scratch roles,
unchanged budgets and accepted-prefill selection. The capability test caught
and fixed a missing explicit count4 check for the FinalFourRows role. The
independent read-only seam review found no material blocker. Strict all-target
pure-direct Clippy (`-j2`, denied warnings), scoped formatting and diff checks
pass on the integrated source. Tests used the exclusive NVMe
`cpu-batch-native-target-20261011` cache; no GPU or service actions were run.
Leaf's sixteen focused quantizer/decoder/shader tests and separate strict
Clippy passed in the leaf owner's source acceptance; they are not GPU evidence.
The controller/leaf are separately owned. No public inference API, installed
service, model catalogue, frontend or default release behavior changes.
