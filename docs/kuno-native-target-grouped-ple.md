# Public native target grouped PLE reads

This default-off source candidate groups the existing completed PLE OLD,
NORMALIZED and NEXT owner maps. It preserves the original numerical and
acceptance contract in [model batching](kuno-native-text-model-batching.md)
and [all-row target output](kuno-native-mtp-all-rows-plan.md). Physical numeric,
lifecycle, closure and timing acceptance remain required. No installed inference,
API, model format, arithmetic, private allocation or allowance changes.

The v158 normal public code receipt records 199 continuations in 23.026516602s
(8.6422103456 tokens/s), 69 verification rounds and 5 native repair chunks.
Draft 4.968547350s, verification 16.263308694s and repair 1.774728855s are separate
controller intervals. The accumulated native phases cover all 74 target chunks:
attention/GR wall intervals 10.832803307s and completed-final proofs 4.249928848s
include repairs. The attention interval can wait for earlier queued expert work.
Process-global submission waits total 9.852846462s across 618 waits, and scoped
mapping totals 2.750658587s. These overlap the phase intervals; they are not
additional elapsed costs.

The actual v158 public language path selects `executor.shared_kernels`, but the
global `next_shared::gr_selected()` stays false outside the count1 laboratory.
PLE's existing three-owner grouping was tied to that unrelated GR arithmetic
selection. Its native target therefore makes three separate full `read_one`
calls per accepted chunk. A separate immutable target policy reuses the existing
`read_full_three` path without enabling global GR arithmetic. This structurally
removes two readback batches per chunk: 148 for the 74 v158 target chunks. It
does not remove any copied bytes or establish a runtime speed gain.

The root-owned public parser/controller may request the policy through these
internal interfaces:

- `Config::reserve_public_grouped_ple_reads()` grants construction permission
  after target rows/latest checkpoint, exactly one public request scope and
  original frozen source/router reservations. It leaves execution off.
- `ResidentModel::set_public_grouped_ple_reads(executor, true, cancel)` selects
  only after accepted prefill, actual frozen router/source activation with
  partition exit and before request I/O opens. CPU13 cold replay remains false;
  its warm selection requires the consumed reset and accepted first token.
  The language selection requires its actual accepted prompt boundary. Once
  selected it cannot be deselected; a fresh owner is required for another run.
- `completed_public_grouped_ple_reads(executor)` returns `(selected, chunks)`
  only at the original completed boundary. Selection requires positive actual
  grouped PLE execution joined to whole-model acceptance. An enqueue, requested
  flag, failed proof or abandoned family commit cannot produce credit.

Selected execution is restricted to original all-row target calls with runtime
count 1..3 and the unchanged tile 8/member 8 projection contract. `Chunk` and
`ThreeGdn` compose in either selection order; count 1 policies/old tile mutation
are rejected before changing policy. Frozen GU/DOWN choices preserve their
existing independent source and arithmetic contracts.

The execution sequence remains first full target packet/fault proof, PLE maps,
full PLE diagnostics/host transition preparation and checkpoint, second same
model fault proof, all-family preflight and commit. The three maps are immutable
completed sources with no intervening writes. Each full owner is 524288 bytes,
so raw maps total 1572864 bytes. Existing CPU capacity plus twice that raw total
plus the original 65536-byte metadata reserve must fit the same 4MiB cap before
mapping. Raw leases wipe/drop before audited cleanup and candidate transition;
the original independent transition-capacity check remains. All finite, bitwise
OLD/NEXT, normalized inactive, physical padding, cancellation, poison, admission,
epoch and full owner checks remain. The second proof is never suppressed or
moved before PLE diagnostics.

Focused CPU tests cover actual three-map decoding at counts 1/2/3, exact bits
including signed zero, every owner tail/nonfinite/inactive rejection, full
raw-plus-decoded cap/overflow, selection prerequisites, target count/purpose,
and actual dispatch/accepted counter parity with no false or overflow credit.
All 34 PLE/model batch tests pass, including the existing two-fault-proof,
poison and all-family commit-order checks. Strict direct-device all-target
Clippy with warnings denied and scoped format/diff checks pass. These do not
establish GPU performance or whole-model physical acceptance.
