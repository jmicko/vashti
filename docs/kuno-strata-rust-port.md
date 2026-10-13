# Strata Rust whole-model port experiment

## Current checkpoint: paused at the user's request, October13

Stopped after the v221 whole-model test. **No further GPU work is authorized
until the user resumes.** The automated goal is paused. Both GPUs and normal
services have been restored, and the bounded sudo refresher is stopped.

Runtime source: `1caf271922abd1dd7aaf3f3d48789e4467362571`, branch
`codex/strata-rust-port`, checkout
`/home/jmicko/.codex/worktrees/kuno-image-speed/vashti`.
Original accepted engine remains on `codex/kuno-native-text` at `6792453`.
Canonical Vashti, installed binaries/defaults, API and frontend are unchanged;
required Vashti changes: **none**. This is a fixed-public laboratory engine,
not a deployed private-input worker. No release acceptance is claimed.

The user requested a coherent Rust port followed by combined execution and bug
fixes, rather than incremental original-CPU13 precision gates. That supersedes
those numerical gates only for this isolated branch/feature. Physical/source
bounds, actual target verification, private erasure, swap/core-disabled GPU
execution and actual recovery still apply.

## Connected implementation

The separate `kuno-strata-rust` target/`strata-rust-port` feature reuses the
already-written Rust GGUF/tokenizer,48-layer Qwen graph, PLE, GDN, QSA and gated
residual operators. Rust and generated WGSL/Vulkan execute inference; no C++,
Strata, llama.cpp or SYCL inference engine is linked.

- One Q8_1 activation plane with F16 effective scale; original IQ2/IQ3/IQ4
  packed target weights; grouped GU/DOWN, cooperative activation staging,
  256-thread32-row workgroups, integer dots, subgroup reductions and FMA.
  Counts1..3 and4 use the same policy.
- Parallel GPU top11 routing, real GPU argmax and compact163888-byte carry
  packet. The complete target vocabulary is still evaluated; logits are not
  copied in full to the CPU. Final real outputs must be finite.
- Q2 sidecar dense BF16/Q6 weights converted to Q8 in Rust/RAM at load. Q2
  experts and IQ4 embedding remain original. No converted model file is saved.
- Strata's106299-token draft vocabulary, sorted without adding/removing IDs;
  only proposal logits are pruned. Full target verification remains authoritative.
- Request-local suffix matching with real first-MTP agreement and measured
  draft acceptance/cost policy. Lookup histories and cache maps are zeroized.
- Adaptive expert tier using frequency/recency and decay0.7, every4 controller
  rounds, at most2 swaps/layer. All resident slots are eligible. Evicted original
  weights stay in the host complement; no disk reread is required for promotion.
- The newest source uses exact persistent bank allocations and192 resident
  experts/layer:96+96 VRAM,144+144+32 host. Both host maps and generated source
  plans derive from shared constants; the original feature-off geometry stays
  144/72. Actual six distinct bank roots are checked against the exact quote.
  Public RAM complement drops from36147609600 to31432704000 logical bytes.
- GPU-driven two-way exchange of original resident/host cells in one joined
  queue transaction. It has no activation/prompt binding, no CPU weight staging
  and no temporary model copy. The new logical bijection publishes only after
  every role/swap completes. Ordinary external-source bindings stay read-only;
  the isolated port's owning transaction alone uses this explicit raw device
  write path. **This newest exchange has not executed successfully on hardware.**
- Intermediate numerical scans use Strata's leaner policy; final checks and
  complete private owner erasure remain. Public runs permit up to512 outputs
  and respect EOS; the original engine still caps this fixture at200.

Pinned upstream: MIT Strata `fb58e0dbc8399662c0e47c76578c6e878b14f6cf`.
`strata-port-NOTICE.txt` and existing ggml/llama.cpp attribution are retained.

## Actual combined results and failures

All measured rates below are **real timed continuations/s**, excluding the first
confirmed token. Fixed public code fixture,temperature0/seed424242, Intel B70.
Different MTP arithmetic/window lengths can produce different greedy trajectories;
these are not an exact same-output Ollama comparison.

| Candidate | Scope | Rate | Evidence |
| --- | --- | --- | --- |
| v205 Q2 | code200 |8.220500|135/189 accepted drafts,23 repairs |
| v205 Q8 | code200 |8.672648|137/183 accepted,21 repairs |
| v207 Q8 | code200 |10.800044|147/155 accepted,5 repairs; no suffix window actually selected |
| v213 Q2 combined | code200 |9.175268|145/161 accepted,7 repairs;144-slot adaptive cache, compact carry, runtime conversion/vocabulary connected |
| v214 Q2 combined | code512 cap,362 emitted at EOS |7.976630|108 rounds,36 repairs;8.823267451s cache adaptation |
| v215 Q2 batched transfers/checks | same cap,362 at EOS |8.458796|42.677466683s timed,6.529365052s adaptation |

v213 cache performed1246 swaps,2549555200B moved,4.467331086s adaptation.
Its resident route fraction was70888/108960 (~65.1%). v214/v215 performed2492
swaps,5099110400B moved. v215 draft5.224239850s,verify24.334071779s,
repair6.570197023s. These completed runs pass internal owner closure and actual
process/GPU/service recovery. **30tokens/s and complete upstream feature parity
are not achieved.** The last successful broader-port artifact is v215 from
`c1b334c3a5e471e6342e5b9cf5900e520704f864`, SHA256
`2d804b91b493466a1987f68b11ad5780e41037ed4937b021731c85c768336f79`.
It uses144 slots and CPU-staged exchanges, not the newest192/direct-exchange code.

Important failures fixed this pass:

- v204 Q2 exact-Q8 admission rejection (`27f14fb`); v204 Q8 receipt permissions
  (`ec21023`). Missing v204 timings must not be inferred.
- v208/v209 shader launch failure hidden behind retained metadata. Router used
  WGSL-reserved `active`; renamed `live` in `9815e85`. Full new shader coverage
  now includes routers, packet and exchange, plus20 projection variants.
- v212 held an outgoing readback lease after wiping it, blocking the sanitizer.
  `32baf8a` explicitly drops it before mutation; v213 then succeeds.
- v216192-slot prototype retained a144-slot whole-model constructor. Connected
  `0f4c623`; no model execution acceptance for v216.
- v217/v219 exposed stale router table extent and metadata budget. `93b4178`
  sizes the table to its bank; `3e09fb6` raises only the isolated port's per-layer
  metadata budget to64KiB/4MiB aggregate. `c627327`/`21ee0c0` preserve original
  failure diagnostics instead of masking them with a poison notification.
- **v221 reached successful cold prefill,432-page host materialization, frozen
  GPU routing and decode startup, then rejected the first four-row DOWN launch.**
  Its four-row launch wrapper still contained two literal72/144 shard tables.
  `d682e04` connects both to the selected source partitions; `1caf271` passes
  source checks. **No GPU rerun after this fix**, honoring the user's pause.
  The direct GPU exchange therefore has no hardware/timing acceptance yet.

The last tested but failed v221 artifact is source `3e09fb6789f1bd92d5c543671b72b2136c32375f`, SHA256
`b6fc683d5065626c8842ea894fb613a4bc3161254b07402507e224f454ffe9a1`, under
`native-strata-port-v221-build-20261013`. Do not select it as a working artifact
or silently repeat it; it lacks the fixed four-row guards.

## Checks and actual stop receipt

- Latest source: Rust port Clippy `-D warnings`, fmt/whitespace, and original
  `kuno-text-lab` direct-device compilation pass. Saved logs:
  `strata-port-final-clippy2.log`, `strata-port-original-check.log`.
- v221 full new-controller WGSL validation (including exchange) plus20 projection
  WGSL/SPIR-V variants passes. A fresh release artifact from latest source still
  needs its own full shader check. Compiler acceptance is not hardware execution.
-1346 original-engine tests/1ignored passed at `f0ed07a`, **not current source**.
  Full regression and representative code/prose runs remain outstanding.
- v221 failure: internal private close reports complete=1; actual cleanup passes.
  Fresh read-only16:29:23UTC receipt `native-strata-port-final-pause-20261013.json`:
  admission idle/unpaused/no latch, all three listeners200, B70used50688000B,
  Nvidia compute0 (679MiB desktop), classified GPU faults all0, unchanged
  boot`b32654b7-a49b-493d-8305-46da15b6eac6`/kernel7.0.0-38-generic.
- Recorded sudo refresher3421970 terminated and exited. No benchmark/compiler
  job or queued GPU work is left. Do not refresh credentials automatically.

All development artifacts/receipts above live under
`/media/jmicko/faststore/kuno-development`. Benchmark logs omit generated text;
fixed public output is displayed live only. No private input path or content
persistence was added.

## Resume only after explicit user instruction

1. Read AGENTS/coordination, this checkpoint and current git status/log. Preserve
   canonical Vashti and inherited `text/__pycache__/`; do not restore the lab to
   main or overwrite newer work. Original accepted engine remains6792453.
2. Build a fresh immutable `kuno-strata-rust` artifact from latest committed
   runtime source, e.g. `python3 apps/kuno/benchmarks/text/build-strata-port.py
   native-strata-port-v222-build-20261013`. Verify SHA/source; run its
   `--validate-port-shaders` and registration-only preset before GPU execution.
3. With the user's GPUs free and cached sudo available, read the Demons shell
   before typing. Run a fresh protected code/Q2 fixture through `run-strata-port.py`.
   It uses MemorySwapMax=0/LimitCORE=0, checks actual idle memory, pauses admission
   and restores/physically verifies services/devices in finally. Never benchmark
   in an ordinary swappable shell, reuse an output directory or reset a GPU.
4. First required acceptance is the complete192-slot/direct-exchange run after
   d682e04. Check that actual exchanges occur, real output is readable, target
   verification and all owner closes pass, and actual memory/faults recover.
   If it fails, preserve the real cause and fix the combined path; don't switch
   back to isolated original-precision kernel trials or claim performance.
5. Only after that works, evaluate initial ranked residency, faster accepted
   prefix rollback and a6/8-row verifier. Those, XMX prefill, compact long-context
   KV and the general private-input worker remain unfinished. The current bank
   count is fixed192, **not auto-sized**. Do not call the whole upstream rewrite
   complete. Request-local adaptive residency is implemented; inference history
   must never be saved in Kuno.
6. Finish appropriate original-engine regressions and representative code/prose
   acceptance before selecting an artifact. Keep API/UI/release work elsewhere.
   On losing sudo, document a clean checkpoint; avoid privilege workarounds.
