# Fixed-public HC phase diagnostic

Internal diagnostic only. Preserve the full coupled experimental bundle and
original correctness/routing/ownership checks. No API or Vashti changes.

The preceding complete public profile reports approximately891ms in combined
GR-attention and GR-router against Strata's differently scoped HC185ms. That
coarse comparison cannot identify which GR operator is expensive. Add six
fenced wall-time substeps within ONE attention GR invocation: normalized R,
down projection,injection projection,low transform,up projection,mixed output.
Use the existing compiled-public authorization plus exactly request3/start0/
count4096/layer24 selection. The original selection geometry CPU test remains.

Only that attention invocation installs a thread-local RAII scope. Error and
normal drops restore its prior value. A new CPU control checks disabled sampling
leaves the scope unchanged. Private and clean calls check the existing disabled
collector first: no environment lookup,no new client lookup,no sync,no content
or resource retention. Existing full arena/scope/alias checks precede timers.
No prompt,route,expert,token or tensor values are recorded; six static labels
publish [calls,completed-wall-ns,enqueue-wall-ns] through the existing collector.
These are completion-fenced wall times,not native GPU timestamp or pure GPU
execution claims. Extra fences mean this diagnostic is NOT a clean timing or
installation selection. Reuse the original frozen correctness/model chain and
verify actual physical recovery. Source only until the matching worker runs.
