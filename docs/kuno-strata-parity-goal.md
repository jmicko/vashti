# Strata parity goal — 2026-10-20

The user explicitly requested a new goal to match Strata. Work remains Kuno-only
on `codex/strata-rust-port`, using the agreed integration032e2cdd and the installed
worker c8db19de / SHAfb31e124 as the accepted baseline. Preserve the canonical
Vashti checkout and uncommitted coordination files. No API/frontend change,
release, reset/reboot, model/quant substitution or relaxed numerical/ownership
checks is part of this goal. Current GPU permission and sudo apply to this run;
stop protected execution at a safe checkpoint if the credential expires.

## Reference and acceptance scope

Pinned MIT upstream Strata v0.1.41, source
`fb58e0dbc8399662c0e47c76578c6e878b14f6cf`, is available for source comparison at
`/home/jmicko/.codex/worktrees/kuno-native-split-gguf/strata-research-20261009`.
The old comparison installation/binary/container was removed at the user's
request. Its original public receipts/configuration are retained under
`/media/jmicko/faststore/kuno-development/strata-comparison-evidence-20261009`.

Historical local Strata uses the same original DASLab IQ3_S shards, Q2 MTP,
Intel B70, context32768, int8 KV and direct PLE; spec4/min-p0.5 and CJK draft
vocabulary. Seed424242/temperature0,35-token merge-code prefix0.7058s and30-token
prose prefix0.6373s;200-output code/prose decode27.3/36.0 tokens/s. Current native
Kuno code uses seed0, permits1024 outputs and produces501 rather than the previous
362 tokens; warm prefix1.4305s and decode32.92 tokens/s. These are useful reference
bounds, **not a new matched sustained comparison**.

The upstream B70 report's4095-token prefix980–1002 tokens/s (~4.1–4.2s) uses PLE
RAM, different input/configuration and an earlier host. Kuno's current4245-token
compiled-public check takes24.9658s, toolbox4207/4293 prefixes29.506/29.920s.
Do not claim parity by comparing unlike workloads, truncating tools/history,
changing the quant or reporting isolated first-token speed. Before declaring
parity, require an agreed matched local upstream workload/settings/timing scope
or a sufficiently documented equivalent reference; distinguish prompt/decode,
load/first-use compilation, final erasure and whole-response time.

## Connected high-potential work

- Matrix: improve compiler-visible K64 vector/stage/MMA execution while retaining
  the accepted paired32KiB shared operands. The prior register-first variant
  materially regressed; do not confuse source prefetch intent with proven speed.
- Native graph: investigate redundant global memory boundaries between command
  segments while preserving original per-node physical-range hazards, crossqueue
  timelines, external entrance/exit, host visibility and final actual join.
- Short prefix: extend the existing genuine MAX3/64 path to connected MAX8,
  no vocabulary head, original device residual ownership and exact known-input
  MTP teaching. Preserve separate real nonce/purpose/capture identities, complete
  target/draft cursors, erasure/rebind and the final ordinary count1 input.

Only compiled-public profiling may retain aggregate expert-member histograms
and arithmetic-work estimates; no identities, routes, text, hashes of content
or private request profiling. Production worker logging/persistence stay unchanged.
Use the real GPU numerical controls before accepting complete generation, then
matched speed, thinking/tools/semantic repeat/privacy, actual unload and physical
recovery. Source isolation does not isolate GPUs: only the root runs shared tests.

Current review ownership: matrix agent603a1de7 owns strata_matrix.rs; runtime
agent owns strata_graph.rs; short agent owns controller/target/MTP/IO plus one
immutable known-input headless recording identity in server/base.rs. Root owns
integration, public profiling geometry, GPU testing and installation.
