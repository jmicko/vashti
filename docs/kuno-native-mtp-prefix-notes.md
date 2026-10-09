# Native MTP cache preparation and accepted-prefix research

This internal Rust preparation adds no backend routing, API, production/private
request path or speculative performance claim. The existing MTP graph remains
the sequential public-fixture reference described in
[the MTP plan](kuno-native-text-mtp-plan.md).

## Implemented narrow seam

`MtpGraph::append_kv(executor, sequence, token, preceding_hidden, cancel)` returns
`Result<()>`. It consumes exactly one teacher-forced token, paired with the
preceding **real target** 10,240-wide residual. It reuses conditioning,
HC-attention mixing, K/V projection, key normalization and positional RoPE from
the full reference step. The unused injection, query, attention readout, MoE
and output head are skipped. No draft logits or residual are returned/retained.

Success advances the cache by one position. Binding, token, extent, finiteness,
health and context preflight failures do not mutate a completed prefix. An
execution failure/cancellation clears the entire cache, resets its position and
poisons it until explicit reset; callers cannot reuse a half-committed prefix.
The full `step` shares the original math and this same transaction rule.

Compiled nonzero synthetic tests compare every actual private K/V bit and
position against full same-input steps over several positions, then compare the
next complete step's outputs. Rewind tests replace provisional draft
conditioning with different target carries and match a fresh full replay.
Failure, malformed output, cancellation and post-cache-publication error tests
exercise clearing/poison/reset. They check 8 cache-only projections versus 51
full-step projections, not physical latency or real-weight parity. The reduced
synthetic fixture does not prove strict downloaded-sidecar admission or an
independent whole-head numerical oracle. Existing scalar helper whole-stack
private error-path erasure remains unaccepted.

## Pinned mechanism reference

Strata revision **`fb58e0dbc8399662c0e47c76578c6e878b14f6cf`**, with the
[MIT license](https://github.com/Niko1221/Strata/blob/fb58e0dbc8399662c0e47c76578c6e878b14f6cf/LICENSE),
is inspected as a mechanism reference; no Strata backend is linked.

- [src/core/mtp.cpp:663–669](https://github.com/Niko1221/Strata/blob/fb58e0dbc8399662c0e47c76578c6e878b14f6cf/src/core/mtp.cpp#L663-L669)
  separates the K/V-only front from the complete draft layer.
- [src/core/mtp.cpp:1048–1050](https://github.com/Niko1221/Strata/blob/fb58e0dbc8399662c0e47c76578c6e878b14f6cf/src/core/mtp.cpp#L1048-L1050)
  catches up K/V for the verification window and executes the rest only for the
  row which makes the next draft;
  [1245–1252](https://github.com/Niko1221/Strata/blob/fb58e0dbc8399662c0e47c76578c6e878b14f6cf/src/core/mtp.cpp#L1245-L1252)
  bounds default catch-up to the accepted prefix.
- [src/core/verify.cpp:1796–1803](https://github.com/Niko1221/Strata/blob/fb58e0dbc8399662c0e47c76578c6e878b14f6cf/src/core/verify.cpp#L1796-L1803)
  retains each layer's raw QKV, prepared recurrence input and controls, then
  replays only accepted recurrence rows from the old committed state. It does
  **not** require a complete recurrent-state checkpoint after every draft.
  [1806–1823](https://github.com/Niko1221/Strata/blob/fb58e0dbc8399662c0e47c76578c6e878b14f6cf/src/core/verify.cpp#L1806-L1823)
  also restores the indexer tail and rebuilds accepted QSA indexer appends, then
  selects the accepted PLE history snapshot.

## Remaining native work

A two-draft target window consumes the confirmed token plus zero, one or two
accepted drafts: commit **one, two or three input rows**, respectively. Native
MAX8 dense projections and GDN chunk kernels already exist, but the current
resident graph publishes a complete chunk and produces only the final row's
head. It still needs all-row target logits/carries and a new all-family
accepted-prefix transaction; an old full-chunk stamp must never be fabricated
for a shorter prefix or used to recover a poisoned ledger.

Device-side GDN prefix replay could retain raw/prepared QKV and controls per
layer, then rerun convolution/recurrence for the selected prefix. This avoids
additional full recurrent checkpoints and does not repeat dense projections or
MoE. The current shared workspace is overwritten by later layers, so dedicated
bounded per-layer saved inputs are necessary. QSA's raw KV/cursors and pooled
indexer tail, plus PLE token/raw-convolution history, need explicit prefix
publication and rejected-suffix cleanup too. This is a proposed owner design,
not implemented device replay or a measured memory/performance result.

Draft KV conditioned on draft-generated hidden state must be repaired with the
verified **target** carries even when its tokens are accepted. Rewind after the
last correctly conditioned cell, append each accepted token with its preceding
target carry, and use the next confirmed correction/bonus token for the next
complete draft step. Prompt cache preparation likewise needs the preceding
target carry at every prompt position. Final-prompt carry alone cannot rebuild
a missing causal draft cache. Draft and target state ownership remain separate.
