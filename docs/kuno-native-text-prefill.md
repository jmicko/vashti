# Native Rust prefill implementation boundary

This is an internal implementation plan for `codex/kuno-native-text`, following
the [exact graph contract](kuno-native-text-graph.md). It changes no public API,
model catalog, scheduler or Vashti consumer. Multi-RHS matrices and isolated
operators are groundwork; repeatedly calling the existing token step is not
batched prefill. See [measured acceptance](kuno-text-performance.md).

## One transaction across the whole model

Use fixed public capacity eight, with actual count 1–8, token IDs, positions,
bases, strides, routes and probabilities passed as runtime data. Only original
model geometry and the public tile capacity may specialize shaders. A chunk
belongs to one graph binding, physical client/stream, arena epoch and sequence;
it cannot combine requests or outlive the owning request. Prompt scratch,
candidate histories and route planners remain transient, guarded and bounded.

Admit every complete allocation page and both independent I/O ledgers before
allocation. Include current and candidate state, both residual batches,
projection/finite-status scratch, route unions, PLE sources and simultaneously
retained expert sources. Do not reuse a main telemetry fence as runtime staging
credit or silently enlarge an allocator/cache limit. Validate all extents,
full physical backing and nonaliasing before unchecked launches.

Run all eight columns through layer zero, then layer one, and so on. At each
layer, preserve position order within its causal operations. Current committed
GDN/QSA/PLE state stays unchanged until the entire chunk has completed, every
finite/status fence has passed, cancellation is still clear, and the physical
context is healthy. No layer commits early. Any failure after mutation poisons
and erases the whole sequence; there is no partial chunk replay or CPU fallback.
Only whole-chunk acceptance advances the cursor by the active count.

For an intermediate prompt chunk, omit the unused final vocabulary head while
still accepting the full residual/finite fence. For the last chunk, execute the
final mixer and head for its last active column only. Generation continues with
the existing single-token path. Prefix reuse is a separate feature; this plan
does not imply persistent private prefix caching.

## Layer operations

**GR:** normalize each original residual column independently with the existing
64-lane reduction tree. Batch the down/up/injection projections using original
packed weights. Preserve the four-stream addition order, SiLU(x/4), injection
weights and original residual during combine. The final output mixer has no
injection/combine. Inactive columns, gaps and padding must remain untouched.

**GDN:** batch QKV, closing gate, alpha and beta projections. Consume raw causal
convolution history in position order, then apply SiLU, modulo head pairing,
L2 Q/K normalization and Q/sqrt128. A recurrence chunk reads committed state,
computes all active outputs in original decay/prediction/delta/readout order,
and returns one candidate final state. Preserve per-head RMS and sigmoid gate
before batched output projection. Candidate raw history advances by exactly the
active count. A chunk recurrence alone does not cover convolution/projections.

**QSA:** batch projections, normalization and original Rust rotary-table reads.
Each query sees only committed KV plus its own and earlier chunk columns.
Pooling includes only complete groups of four raw indexer keys, before
normalization and RoPE at the first block position. A group completed by a later
column is invisible to earlier queries. Preserve exact incomplete tails,
deterministic block order and head/12 association. The candidate selector cache
and visible cursor must not expose future columns. Test across 2048/2049/2052
and group boundaries before accepting batched attention; a mask over a fully
pooled future chunk is insufficient.

**PLE:** derive each token's rows from original token IDs and the candidate
history in position order, preserving the separate boundary token and all
sixteen u64 hash branches. Read original lexical rows in bounded slices; selected
rows and their decoded embeddings are private source data. Batch projections
where valid, but consume normalized gated-value history at lags 9/6/3/0 in order.
Commit token and convolution histories only with the whole chunk.

**MoE:** keep global router softmax and every token's top ten, renormalization,
clamp and separate shared expert. An eight-token union can contain eighty
experts, exceeding the legacy 64/72 slots. Plan bounded waves before uploads;
protect each wave's experts through completed consumer use. Preserve each
token's original global probability/rank order in output reduction regardless
of grouping or wave order. Batch vectors that use the same original expert,
without compiling expert IDs or token counts into pipelines. A partially filled
or executed bank invalidates the whole transaction, even if its old host plan
was never committed. Do not substitute fewer experts to fit the cache.

### Expert waves and launch cost

Start with contiguous token waves: add each token's ten distinct experts to
the protected union until the next token would exceed the configured slots,
then start a new wave. The minimum ten-slot bank always permits one complete
token. Preserve each token's rank/probability association separately from the
union's residency order. All eighty selections must be accounted for exactly
once even when several tokens share an expert. A 96-slot bank can hold the
maximum eight-token union, but its larger allocation and segmented dispatch
remain a measured choice rather than a prefill prerequisite.

A wave plan protects its complete union before choosing any missing victims.
Populate banks in bounded source windows; completing original public copies
can retire temporary sources, but cannot release protected bank entries or
commit private causal state. Include every window's complete source pages in
admission. Keep the bank plan and original matrices alive until all consumers
have completed. A partial fill or dispatch invalidates the entire chunk.

For repeated experts, gather their participating input columns using runtime
token/rank maps, reuse original packed weight decoding across those columns,
and scatter down outputs back to their original token/rank partials. Never
compile expert IDs or compacted token maps into pipelines. Inactive compacted
columns remain bounded by the fixed eight-column allocation. Reduce ranks0..9
in the original order only after every wave has supplied its partials; keep
the separate shared expert and scaling expression unchanged.

Measure launch costs before choosing a grouping strategy. Launching separate
gate/up/down matrices for every unique expert can cost up to eighty sets of
dispatches per chunk and erase an arithmetic benefit. Compare a fused grouped
bank kernel against that simpler reference, with runtime group maps/counts,
original per-column trees and immutable shard geometry. The already accepted
isolated multi-RHS matrix win does not establish that this routing arrangement
will accelerate real prefill.

## Acceptance before connection or selection

1. Independent host geometry/overflow/budget/ownership checks and offline shader
   validation, including invalid control, nonfinite, stale epoch and failure paths.
2. Both physical GPUs: every active column against existing single-token GPU
   execution and independent core operators; odd and actual model geometry,
   counts 1/3/5/8, irregular bases/strides, inactive tails and full-page padding.
   Inspect full owned main spans and tracked cleanup stages before success.
3. Causal layer fixtures with nonzero starting state, chunk partitions, complete
   block/tail/PLE-history boundaries and errors between candidate writes. Verify
   committed state never changes before the final accepted fence. Cancellation
   destroys all candidate/current request ownership and fresh-owner parity returns.
4. Whole-model public fixtures: single-token versus chunked execution under the
   same original weights/options, then the exact committed 75/80-token prompts
   and 128-token generation. Establish a documented quality/rounding criterion
   before accepting longer outputs that differ from the scalar/Ollama references.
5. Compiler-idle paired measurements with raw ordered samples: cached operator
   timing first, then complete model prefill and generation. Record OS cache
   policy, model/process residency, lazy loading, actual memory/device state and
   executable identity. An operator gain is not a whole-engine or Ollama speedup.

Production request ownership, worker teardown/recovery, long contexts, supported
sampling/tool/vision behavior and managed native/`ollama.` routing remain separate
acceptance requirements. Nothing here authorizes a production route or release.
