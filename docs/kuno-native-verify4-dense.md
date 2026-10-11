# Fixed four-row original-packed dense projection leaf

This internal default-unused laboratory leaf adds a separate fixed count4
projection and finite-projection entry. It does not activate a four-row target
transaction, a third MTP draft, router/GDN/QSA changes, or a public selector.
Existing count1/scalar and Verify3 sources, helpers, launchers and defaults
remain unchanged. Source checks do not establish numerical device acceptance
or a throughput gain.

## Exact scope and ABI

`next_shared::matrix_verify4` and `matrix_verify4_finite` require exactly four
real RHS rows in the original `BatchSpans`. All ten existing dense formats
remain original: F32, F16, BF16, Q5_0, Q8_0, Q4_K, Q5_K, Q6_K, IQ4_NL and
IQ4_XS. Activations remain F32. IQ4 decoder selection must be Original; the
separate register experiment is rejected rather than silently selected.

The fixed shader uses a vec4 of independent RHS partials, the same original
packed helpers, factor/dot expressions, column order and subgroupAdd reduction
as Verify3. Its first three component expressions are unchanged. The fourth
uses the original dot at input offset three times the original RHS stride.
128 physical threads and subgroups of16 or32 remain required; each subgroup
still produces one output feature. Output-tile grid X remains ceil(rows/4),
with no token dimension. Runtime offsets and complete owner sizes remain INFO
values, never shader/cache keys. MatrixVerify4 and MatrixVerify4Finite have
separate immutable kernel identities and names from all original kernels.

Bindings remain weight, full input, full output, original one-word fault and
one INFO. Normal INFO is32bytes and finite INFO48bytes; each consumes the
existing one2048-byte owned ledger record. No GPU owner, scratch/workgroup
array, extra dispatch, cap or allocation reservation is introduced. Existing
MAX8 roots already admit four RHS rows; the parent must continue proving full
physical source identity, disjoint complete roots, padding, clears, erasure
and unchanged all-family/carry/greedy acceptance. The leaf authenticates spans,
not their source provenance.

Count4 is checked on host before dispatch and uniformly in the shader before
weight/input/output access. The same subtraction/division bounds avoid offset
wraparound; malformed spans set the original sticky fault bit and return.
The full finite variant retains the existing complete matrix-interval guard,
active self-checks and last-tile logical complement scan. It never reads an
active interval that another weight tile may still be writing. Logical
bindings remain at most64MiB, tail scan grid X at most4096, and full physical
padding/status/final-fault proofs remain independent parent obligations.

## Required acceptance and limitations

Focused checks must cover fixed-count rejection, exact spans and partial
owners, both subgroup widths, four-row output coverage including MAX8 gaps,
all original packed producer expressions, Checked/Unchecked actual source and
SPIR-V with runtime workgroup zero-initialization None, complete finite
complement and sticky malformed/NaN faults. Existing Verify3 checks are retained.
Per-component CPU replay provides source arithmetic evidence, not GPU output.

A later parent must implement real count4 transactions and target/router/MTP
state/capacity prerequisites before selecting this leaf. That work must retain
unchanged original numerical tolerances, full CPU13 vectors, causal carries,
full vocabulary greedy verification, real rejection/repair and complete owner
close/recovery gates. Three real drafts and four target rows may reduce round
count when actually accepted; draft cost and acceptance remain unmeasured.
No connected count4 speed or numerical acceptance is claimed by this leaf.

## Source checkpoint

Nine new leaf tests and30 existing shared/Verify3 operator tests pass (39 unique
checks). Actual Checked/Unchecked sources for ten formats, widths256/2560 and
plain/nonfinal-finite/final-finite entries produce120 validated SPIR-V variants
with workgroup zero-initialization None. CPU F32/Q8 replay checks all four RHS
components at both subgroup widths without asserting a driver reduction order.
Original Verify3 builders/validators/launch bodies and every prior Kernel hash
payload are preserved. Strict direct-device all-target Clippy with warnings
denied, scoped formatting and whitespace checks pass. Checks used the standalone
native-text manifest, `--no-default-features --features direct-device`, `-j2`
and only the existing cpu-batch-native-target-20261011 cache. No GPU, service,
model allocation or physical validation was performed. Future parent/physical
acceptance remains required; this leaf has no connected runtime selection.
