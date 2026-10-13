# Strata Rust whole-model port experiment

User authorization, October13: build a complete model execution candidate first,
then run it and fix errors, instead of gating each optimization on original
CPU13 precision. This supersedes that restriction **only for this separate
experimental branch/feature**. The normal native engine and deployed Kuno retain
their existing contracts.

Upstream is MIT Strata `fb58e0dbc8399662c0e47c76578c6e878b14f6cf`.
`strata-port-NOTICE.txt` includes its license; the existing IQ notice includes
ggml/llama.cpp attribution. The port is Rust plus Rust-generated WGSL/Vulkan;
no Strata, llama.cpp, CUDA or SYCL engine is linked into inference.

The standalone `kuno-strata-rust` target uses the existing Rust GGUF/tokenizer,
48-layer Qwen graph, PLE hashing/grouped reads, GDN recurrence/convolutions,
QSA compression/selection, gated residuals and model ownership foundations.
It ports Strata's coupled execution policy across the whole target: one Q8_1
activation plane with F16 scale, original IQ2/IQ3/IQ4 packed weights, grouped
GU/DOWN, cooperative activation tiles,256-thread32-row workgroups, integer dots,
subgroup reduction, FMA, GPU source plans and direct mapped-host weight reads.
Regular counts1..3 and count4 use the same policy. Plans/audits match the new
20 GU/80 DOWN grid. Physical spans, device limits and private erasure still hold.

The full Q8 or original Strata Q2 drafter runs on B70. Four-token target
verification accepts only matching target predictions. In-request suffix lookup
can avoid full MTP forward/head work; accepted real target carries repair the
draft KV cells. No forced reference tokens, substitute experts or persistent
request history. Approximate vector drift is recorded rather than rejected by
the previous two-plane CPU13 threshold; finite values and target acceptance
remain checked.

This reuses already ported Rust operators rather than duplicating them. It is a
whole-model prototype, **not a claim that every upstream feature is implemented**:
Strata's XMX prefill, runtime Q8-repacked/CJK-pruned sidecar, auto-sized/LRU expert
cache and6/8-token verifier are not yet included. No API/UI/install/release change.
The older engine is preserved at `6792453` on `codex/kuno-native-text`.

Build and runner are in `apps/kuno/benchmarks/text/build-strata-port.py` and
`run-strata-port.py`. Runner uses only fixed public code/prose fixtures, seed424242,
temperature0, a swap/core-disabled cgroup and an actual idle GPU check. Generated
public text is displayed live and excluded from saved logs/receipts. Numeric
records include actual per-member-count dispatch credit, source/artifact identity,
MTP/lookup accounting, throughput, owner closures and recovered host state.
Private inputs have no route into this prototype. Runtime state is transient.

Status: implementation written as a single pass; compilation and whole-model
acceptance are next. Existing per-leaf numeric suites are intentionally deferred.
