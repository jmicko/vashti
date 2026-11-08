# Selected Intel QSA active-prefix scoring — 2026-10-20

This is an internal candidate, not a new API, installed version, or measured
speed result. It keeps the original QSA owners, full-capacity strides, sticky
fault, causal selection, accepted cache state, and request erasure unchanged.

Pinned MIT Strata `fb58e0dbc8399662c0e47c76578c6e878b14f6cf` selects the normal
Intel/SYCL attention fallback unless `STRATA_PROMPT_ATTN_XMX` is explicitly set:
`sycl/src/kernels/cuda/qsa_prompt_attn_xmx.dp.cpp:364–368`. The selected fallback
still batches attention32 queries at a time. The default GDN recurrent kernel
also already has the same CB32, four-block/value-head software-pipelined shape
as the Rust path. An inactive CUDA tensor implementation is not the explanation
for the remaining Intel attention gap.

There is a concrete selected-path difference in index scoring. Upstream
`sycl/src/prefill/prefill.cpp:2986–3000` passes the last query's `n_bid+1` as the
active reach for each256-query scoring/selection batch.
`sycl/src/kernels/cuda/qsa_select.dp.cpp:1615–1630` clamps the score grid to that
reach. The original Rust whole-prompt path launches against the full context
capacity for every batch, despite each extra block returning without arithmetic.

The candidate computes the same canonical reach from the actual batch start,
count, and admitted context block capacity: `min((start+count)/4+1, blocks)`.
It narrows only the prompt score dispatch. Both score and flag rows retain their
original full `blocks` stride and all original buffer handles/provenance.
The prompt-only selection entry requires the same canonical reach; the GPU
validator and selector reject any query whose causal visible blocks exceed it.
Validation still visits every visible score/status. Radix selection still skips
the exact-four zero-weight tail before reading its score, and never reads future
or omitted slots. Thus no stale omitted score, including a large finite value
or NaN, can become a favorable candidate. Clearing all full-capacity score rows
would add unnecessary memory work and is not needed for this ownership contract.

Every unused selected-output tail is explicitly overwritten with the out-of-
context sentinel `blocks*4`. Those writes are disjoint from the live selected
interval, so they need no extra dispatch or storage barrier. The ordinary
MAX8/decode selector retains its full-capacity bound: its native graph Position
is rebound on replay, so using a fixed first-recording active reach there would
incorrectly reject later positions. No capture authority or dynamic-parameter
guard is loosened.

For the first4096-token chunk at context32768, the score grid drops from16,384
to1,104 CTAs per query-batch aggregate, or93.26% fewer empty CTAs. This is not a
93.26% attention speedup: those omitted CTAs already return early, and the QSA
profile includes indexer, scoring, selection, sparse attention, and merging.
The previous public profile attributed3.001s to that combined family, out of
6.329s total attention. Finer timing and actual complete-request correctness are
still required before interpreting the performance benefit.

CPU evidence: three tests compile the actual pure geometry helper and actual
selection source generator using existing dependency rlibs, covering partial
and exact-four tails, start/context boundaries, overflow, full owning strides,
canonical grid count, unread score padding, and initialized selected tails.
The actual DEAD/VALIDATE/SELECT WGSL all pass the existing CPU Naga validator.
No GPU, service, sudo, or shared Cargo build was used for this preparation.
Actual long/short/repeated numerical, API, erasure and physical-recovery checks
remain parent-owned hardware gates.
