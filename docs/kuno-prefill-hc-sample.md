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

## Completed diagnostic8b1fd224

Worker de3ddfd121e29817b79f27cfbf226e40c6742faed9860a99cafa3990bdfb1ba4
passes276CPU/15runtime/all original shader layouts and the original fixed-public
profile. Original routing[7663,26673,26673],correct fixed result and physical
recovery pass. Exactly one sample per six keys,on request3/layer24/count4096;
first two requests emit none. No new full457operator claim for this diagnostic
binary; the unchanged arithmetic bundle has separate complete5ae19f5e evidence.

Completion-fenced milliseconds: norm1.503173,down1.205360,injection0.990169,
low0.422143,up1.362712,mix1.169040. Sum6.652597ms. These include per-substep
submission/completion costs and cannot be extrapolated as pure GPU time.
Diagnostic repeated prompt8.556387362s is NOT clean selection.

Scope correction from pinned Strata source: sycl/src/prefill/prefill.cpp:4236
performs fused gr_write_norm before the next kPfHc marker around2779; therefore
its reported HC185ms excludes much of normalization that Kuno includes. Do not
present185ms versus816ms as equal operations. Its AMD-only fused down/injection
implementation at src/prefill/wmma_gemm.cu:480-574 uses[Wd;Wi]oneGEMM with separate
real weight roots and split outputs. That motivates a Rust native fusion using
Kuno's unchanged chronological K16 accumulation,not copying different BK64 math.

Evidence: hc-phases-diagnostic-public-summary-20261023.json and
hc-phases-diagnostic-connected-20261023.log under the development artifact root.
No installed/config/API/frontend/release/reset/reboot change.
