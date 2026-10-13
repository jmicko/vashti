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

## Current checkpoint

The initial coherent prototype is written and built. It is not a completed
rewrite of every Strata subsystem. Source `f0ed07a` fixes a shader-cache identity
collision found by final checks: `.info` replaces its previous value, so the
identity must contain BOTH the original variant/dtype and the port mode in one
tuple. Never revert that fix. The smaller-batch DOWN transform and original
Cargo target registration are also fixed.

The final frozen v204 Rust artifact SHA256 is
`2964f4b83814f0f39b26afab63410db03d05a6c597818f408f08be4bf9616b33`.
Its20 generated variants pass full WGSL validation and SPIR-V lowering;
metadata/workgroup dimensions agree. Both whole Q8 and Q2 presets pass a pure
registration check with explicitly synthetic memory fields and no GPU created.
Strict port Clippy passes. A single final regression check of the original
engine, after fixing discovered bugs, passes1346 tests with one ignored.
These are source/compiler checks, not GPU execution or model-quality evidence.

**No port GPU benchmark has run and no throughput is claimed.** Sudo expired
during implementation; one `sudo -v` prompt is waiting in the shared shell.
The user was asked once. Do not repeatedly ask or weaken the swap/core/admission
requirements to bypass it. No models were loaded and no services changed.
Fresh unprivileged checkpoint confirms all three HTTP listeners200, actual
B70used50688000B and Nvidia compute0.

Do not use v199–v202 artifacts: they predate the cache-identity fix. v204 is the
artifact to test. Original native code is preserved at6792453 on its branch.
Native30tokens/s and full upstream feature coverage remain unachieved.


## Exact first full-model run after sudo is restored

Read the shared shell first. When its authentication prompt completes, start
`keep-sudo-active.py` from that same terminal with `--minutes240` and a pid file
under the development base. The helper refreshes only an already-authorized
cache, never asks for or reads a password, and exits after four hours or a failed
refresh. Stop its recorded PID when ending the work; do not leave it behind.
This addresses the missed refresh during the initial broad implementation.

From the root checkout in that terminal, use:

```sh
sudo -n python3 apps/kuno/benchmarks/text/run-strata-port.py native-strata-port-v204-build-20261013 native-strata-port-v204-code-q2-intel-20261013 code q2
```

The last path is fresh (the earlier failed sudo attempt never launched a run).
The launcher pauses admission, captures fresh actual Intel memory, runs the
fixed public code fixture in a swap/core-disabled cgroup, displays public text
without saving it, records numeric results, and restores admission in finally.
If Q2 encounters a runtime bug, retain its failure and try a fresh `code q8`
control before changing target math. Once the whole path runs, inspect actual
public output, target verification/repair, per-member-count GU/DOWN dispatches,
throughput and owner/GPU recovery. Fix actual errors; do not resume the old
per-leaf original CPU13 gate for this approximate experiment.

The metadata/source helpers are not the engine: model execution remains Rust
and WGSL. CLI input is still limited to these public fixtures; general private
input/API integration and long-context KV remain separate work. The current
cache capacity, graph/prefill implementations and4-token verifier reuse the
existing Rust foundations and differ from the full upstream runtime. Preserve
those limitations when reporting performance.
