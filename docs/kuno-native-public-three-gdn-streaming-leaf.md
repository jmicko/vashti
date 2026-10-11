# Explicit public three-token GDN streaming leaf

This source-only experiment extends the already explicit recurrence policy with
`RecurrencePolicy::ThreeStreaming`, label `three-streaming`, and the same-signature
`gdn_three_streaming` checked/unchecked entry. Default `Chunk` and the separate
count1 `SingleStreaming` are unchanged. `validate_count` admits only 1..3; the
kernel repeats that bound before storage access. Count and all 20 address
bases/strides remain runtime scalars. Only original GdnShape specializes shaders.
Parent integration owns the fixed-public selector, completed boundary, actual
policy receipt and physical acceptance. Selection must follow original normal
prefill; it cannot derive a shader policy from a private count or alter cold
prefill. No controller, probe selector, protected launcher or API change is
included in this leaf.

Each lane still owns one head/value column. Token0 reads immutable old state;
tokens1/2 read that lane's candidate next column. Each complete ascending-key
prediction pass precedes the ascending-key update/readout pass. Every active
column retains original `decayed = previous * decay`, prediction addition,
`delta = (value - predicted) * beta`, `updated = decayed + key * delta`, and
readout addition in the original i=0..head_dim-1 order, without FMA, regrouping
or tree changes. No 128-element private column array is retained.

A per-lane `next_written` flag stays false until a complete update has written
every candidate cell. If token0 fails before update, later invalid predictions
continue to use old state rather than uninitialized next cells. Validity stays
monotonic across tokens. Any invalid control, state, prediction, delta, update or
readout zeros the final candidate column and every earlier active output/status
row in that column. Old remains unchanged. Other columns, inactive rows, gaps
and physical padding retain their original handling.

The admitted MAX8 owner extents, eight storage bindings, 21 runtime scalars,
64-thread grid, original spans, nonaliasing and same-stream source proof remain
unchanged. No owners, reservations, shared scratch or barriers are added. The
existing enqueue path still validates shape, count, full spans/bindings and
cancellation before launch; deferred status/full-state/owner acceptance remains
the parent's mandatory whole-chunk gate. A candidate next-state buffer is never
committed by this operator.

The proposed benefit is lower private register/spill pressure in the operative
`gdn_chunk` path, whose dynamically indexed local column has 128 f32 cells.
Spilling is a source hypothesis, not disassembly evidence. Streaming adds global
candidate-state reads/writes between tokens and may be slower if the driver
already keeps the original column efficiently. CPU/SIMT/shader checks establish
arithmetic and source constraints only; original full-model verification and
compiler-idle same-artifact language timing are still required before selection.

All 14 `gdn_chunk_probe::tests` passed, including the four new focused tests and
existing Chunk/SingleStreaming regressions. New tests compare exact f32 bits
against original chunk and repeated original scalar GPU-reference arithmetic for
counts1/2/3 and head dimensions7/31/128, plus a tolerance comparison against the
independent Rust core. No C oracle is claimed. Failure injection at every token
checks whole-column/earlier-output erasure and preservation of old, inactive
rows, gaps and physical padding. Actual Checked/Unchecked WGSL and SPIR-V with
workgroup initialization None confirm original binding/scalar ABI and float
operation multiset, no local array/private/workgroup state, FMA or barriers.
The original shader remains a positive control for the removed column array.

Strict main-and-test Clippy passes with warnings denied; Rust formatting and
diff checks pass. All CPU jobs stopped by 2026-10-11 14:31:26 UTC before the
parent's next timed trial. No GPU, services or installed runtime are used by
this source leaf. Physical model parity, complete lifecycle and speed remain
unverified.
