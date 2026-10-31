# Kuno worktree disposition — 2026-10-31

Only the `kuno-level-zero-tiled/vashti` checkout is needed for current native-engine source work.
`codex/native-level-zero-interop-public-20261027` is the single integration branch.
The main checkout remains the live Vashti/application workspace and was not switched.
The retained-routing app contributor is already integrated by equivalent cherry-picks; no merge is needed.

## Uncommitted change disposition

- `codex/strata-rust-port`: committed all accumulated coordination and research notes in `b5b4290c`.
- `codex/kuno-image-performance`: committed the original untracked coordination guidance in `29bd64bf`.
- Discarded only 37 verified generated CPython bytecode files from the old text benchmark cache.
- Integration checkout: the final handoff, condensed guide, preserved history and inventory are committed together.
- Main checkout: `.codex/config.toml` and `demons.toml` are unrelated local/user configuration edits; deliberately preserved, neither committed nor discarded.
- Existing ignored private data, model/runtime evidence, build directories, all local branches and the existing stash were preserved.

## Remaining physical worktrees

Except for the integration and main checkout, these are obsolete as source workspaces.
They remain because they contain physical ignored build/data directories, not unique uncommitted source.
Do not recursively delete them or shared artifacts to simplify the branch list.

| Checkout | Branch | Purpose |
| --- | --- | --- |
| `/media/jmicko/faststore/dev/vashti` | `codex/image-flow-refinements` | KEEP: live Vashti/app and local configuration |
| `/home/jmicko/.codex/worktrees/kuno-comfy-speed-review/vashti` | `codex/kuno-image-whole-rope-residual` | ARCHIVE: physical ignored build directory; source branch preserved |
| `/home/jmicko/.codex/worktrees/kuno-convrot-local-review/vashti` | `codex/kuno-convrot-local-review` | ARCHIVE: physical ignored build directory; source branch preserved |
| `/home/jmicko/.codex/worktrees/kuno-device-route-batch/vashti` | `codex/public-exact-vram-bank-plan` | ARCHIVE: physical ignored build directory; source branch preserved |
| `/home/jmicko/.codex/worktrees/kuno-image-gap-review/vashti` | `codex/kuno-image-contiguous-kv` | ARCHIVE: physical ignored build directory; source branch preserved |
| `/home/jmicko/.codex/worktrees/kuno-image-native-residency/vashti` | `codex/kuno-image-native-residency` | ARCHIVE: physical ignored build directory; source branch preserved |
| `/home/jmicko/.codex/worktrees/kuno-image-performance/vashti` | `codex/kuno-image-performance` | ARCHIVE: physical ignored build directory; source branch preserved |
| `/home/jmicko/.codex/worktrees/kuno-image-residency/vashti` | `codex/kuno-image-residency` | ARCHIVE: physical ignored build directory; source branch preserved |
| `/home/jmicko/.codex/worktrees/kuno-image-speed/vashti` | `codex/strata-rust-port` | ARCHIVE: committed research/history, private/build workspace; not integration |
| `/home/jmicko/.codex/worktrees/kuno-image-worker-residency/vashti` | `codex/kuno-image-worker-residency` | ARCHIVE: physical ignored build directory; source branch preserved |
| `/home/jmicko/.codex/worktrees/kuno-level-zero-tiled/vashti` | `codex/native-level-zero-interop-public-20261027` | KEEP: current engine integration source |
| `/home/jmicko/.codex/worktrees/kuno-native-routing-bridge/vashti` | `codex/native-private-routing-bridge-v160` | ARCHIVE: physical ignored build directory; source branch preserved |
| `/home/jmicko/.codex/worktrees/kuno-native-split-gguf/vashti` | `codex/kuno-live-member-iq-probe` | ARCHIVE: physical ignored build directory; source branch preserved |
| `/home/jmicko/.codex/worktrees/kuno-segmented-iq-moe-probe/vashti` | `codex/kuno-public-existing-expert-fill` | ARCHIVE: physical ignored build directory; source branch preserved |
| `/home/jmicko/.codex/worktrees/kuno-strata-mtp-rust/vashti` | `codex/frozen-source-inactive-workgroups` | ARCHIVE: physical ignored build directory; source branch preserved |
| `/home/jmicko/.codex/worktrees/kuno-text-loader/vashti` | `codex/kuno-native-numeric-validator` | ARCHIVE: physical ignored build directory; source branch preserved |
| `/home/jmicko/.codex/worktrees/kuno-unsloth-quant-support/vashti` | `codex/kuno-grouped-live-reduction` | ARCHIVE: physical ignored build directory; source branch preserved |

## Removed obsolete checkouts

All checkouts below were clean, had no ignored payload, and had no observed user-process working directory inside them.
No force-removal was used. Branches remain; detached tips were given archival tags before removal.
Recreate a checkout from its recorded branch/tag if historical inspection is needed.

| Removed checkout | Preserved branch or tag |
| --- | --- |
| `/home/jmicko/.codex/worktrees/kuno-cli-separation/vashti` | `codex/kuno-native-single-dense-validator` |
| `/home/jmicko/.codex/worktrees/kuno-compact-shuffle/vashti` | `codex/kuno-compact-shuffle` |
| `/home/jmicko/.codex/worktrees/kuno-constant-array-probe/vashti` | `codex/kuno-mtp-batched-repair` |
| `/home/jmicko/.codex/worktrees/kuno-cpu-gpu-stage/vashti` | `codex/kuno-exact-tree-shuffle` |
| `/home/jmicko/.codex/worktrees/kuno-down-staged-planes/vashti` | `codex/kuno-down-staged-planes` |
| `/home/jmicko/.codex/worktrees/kuno-frozen-gu-integer-parent/vashti` | `codex/kuno-frozen-gu-integer-parent` |
| `/home/jmicko/.codex/worktrees/kuno-image-residency-placement/vashti` | `codex/kuno-image-residency-placement` |
| `/home/jmicko/.codex/worktrees/kuno-iq4-down-scales/vashti` | `codex/kuno-english-pruned-q8-head` |
| `/home/jmicko/.codex/worktrees/kuno-large-cohorts/vashti` | `kuno-handoff-20261031/kuno-large-cohorts` |
| `/home/jmicko/.codex/worktrees/kuno-large-packed-tiles/vashti` | `codex/native-large-lookahead-20261027` |
| `/home/jmicko/.codex/worktrees/kuno-layer-gather/vashti` | `kuno-handoff-20261031/kuno-layer-gather` |
| `/home/jmicko/.codex/worktrees/kuno-live-member-integration/vashti` | `codex/kuno-live-member-integration` |
| `/home/jmicko/.codex/worktrees/kuno-native-code-cache-backend/vashti` | `codex/native-public-code-cache-backend-v160` |
| `/home/jmicko/.codex/worktrees/kuno-native-code-cache-key/vashti` | `codex/native-public-code-cache-key-v160r1` |
| `/home/jmicko/.codex/worktrees/kuno-native-code-cache-plan/vashti` | `codex/native-public-code-cache-plan-v160` |
| `/home/jmicko/.codex/worktrees/kuno-native-code-cache-privacy/vashti` | `codex/native-public-code-cache-privacy-v160` |
| `/home/jmicko/.codex/worktrees/kuno-native-gdn-policy/vashti` | `codex/native-gdn-matched-policy-v153` |
| `/home/jmicko/.codex/worktrees/kuno-native-gu-fusion/vashti` | `codex/native-gu-publication-fusion-v155` |
| `/home/jmicko/.codex/worktrees/kuno-native-lazy-driver/vashti` | `codex/native-lazy-driver-objects-v158` |
| `/home/jmicko/.codex/worktrees/kuno-native-local-device/vashti` | `codex/native-local-device-v160` |
| `/home/jmicko/.codex/worktrees/kuno-native-mtp-state/vashti` | `codex/native-mtp-public-state-v157` |
| `/home/jmicko/.codex/worktrees/kuno-native-prepared-pipeline/vashti` | `codex/native-prepared-pipeline` |
| `/home/jmicko/.codex/worktrees/kuno-native-public-code-startup/vashti` | `codex/native-public-code-startup-v160` |
| `/home/jmicko/.codex/worktrees/kuno-native-q3-owner/vashti` | `codex/kuno-public-mtp-gpu-probe` |
| `/home/jmicko/.codex/worktrees/kuno-native-qsa-fusion-plan/vashti` | `codex/native-qsa-fusion-plan-v159` |
| `/home/jmicko/.codex/worktrees/kuno-native-qsa-reference-policy/vashti` | `codex/native-qsa-reference-policy-v159` |
| `/home/jmicko/.codex/worktrees/kuno-native-qsa-transaction/vashti` | `codex/native-qsa-transaction-v159` |
| `/home/jmicko/.codex/worktrees/kuno-native-queued-abort/vashti` | `codex/native-queued-abort-v157` |
| `/home/jmicko/.codex/worktrees/kuno-native-queued-qsa/vashti` | `codex/native-queued-qsa-v159` |
| `/home/jmicko/.codex/worktrees/kuno-native-queued-successor/vashti` | `codex/native-copy-source-audit-v157` |
| `/home/jmicko/.codex/worktrees/kuno-native-retained-backend-plan/vashti` | `codex/native-retained-foreign-backend-plan-v160` |
| `/home/jmicko/.codex/worktrees/kuno-native-retained-routing-app/vashti` | `codex/native-retained-routing-app-v161` |
| `/home/jmicko/.codex/worktrees/kuno-native-retained-routing-backend/vashti` | `codex/native-retained-routing-io-backend-v160` |
| `/home/jmicko/.codex/worktrees/kuno-native-routing-host-audit/vashti` | `codex/native-routing-host-audit-v161` |
| `/home/jmicko/.codex/worktrees/kuno-native-routing-io-app-plan/vashti` | `codex/native-routing-io-app-plan-v160` |
| `/home/jmicko/.codex/worktrees/kuno-native-shared-epoch/vashti` | `codex/native-shared-epoch-v154` |
| `/home/jmicko/.codex/worktrees/kuno-native-state-digest/vashti` | `codex/native-public-state-digest-v157` |
| `/home/jmicko/.codex/worktrees/kuno-native-state-oracle/vashti` | `codex/native-public-state-oracle-v157` |
| `/home/jmicko/.codex/worktrees/kuno-native-state-scan/vashti` | `codex/native-public-state-scan-v157` |
| `/home/jmicko/.codex/worktrees/kuno-native-successor-backend/vashti` | `codex/native-successor-backend-v157` |
| `/home/jmicko/.codex/worktrees/kuno-packed-audit/vashti` | `codex/native-level-zero-public-prototype-20261027` |
| `/home/jmicko/.codex/worktrees/kuno-packed-matrices/vashti` | `codex/level-zero-tiled-shader-20261027` |
| `/home/jmicko/.codex/worktrees/kuno-packed-runtime/vashti` | `codex/native-level-zero-interop-design-20261027` |
| `/home/jmicko/.codex/worktrees/kuno-prefill-backend-audit/vashti` | `kuno-handoff-20261031/kuno-prefill-backend-audit` |
| `/home/jmicko/.codex/worktrees/kuno-prefill-device-validation/vashti` | `codex/prefill-native-host-command-batching-20261020` |
| `/home/jmicko/.codex/worktrees/kuno-prefill-flow-audit/vashti` | `kuno-handoff-20261031/kuno-prefill-flow-audit` |
| `/home/jmicko/.codex/worktrees/kuno-prefill-matrix-audit/vashti` | `codex/native-routed-wide-probe-20261020` |
| `/home/jmicko/.codex/worktrees/kuno-prefill-transfer-review/vashti` | `codex/prefill-expert-piece-grammar-20261020` |
| `/home/jmicko/.codex/worktrees/kuno-profile-gemm-tiers/vashti` | `kuno-handoff-20261031/kuno-profile-gemm-tiers` |
| `/home/jmicko/.codex/worktrees/kuno-public-language-lab/vashti` | `codex/kuno-frozen-partition-exit-public` |
| `/home/jmicko/.codex/worktrees/kuno-public-read-schedule-guard/vashti` | `kuno-handoff-20261031/kuno-public-read-schedule-guard` |
| `/home/jmicko/.codex/worktrees/kuno-regular-four-cell/vashti` | `kuno-handoff-20261031/kuno-regular-four-cell` |
| `/home/jmicko/.codex/worktrees/kuno-shell-export/vashti` | `codex/kuno-shell-export` |
| `/home/jmicko/.codex/worktrees/kuno-single-q8-gu-probe/vashti` | `codex/kuno-single-q8-gu-probe` |
| `/home/jmicko/.codex/worktrees/kuno-target-grouped-ple/vashti` | `codex/kuno-target-grouped-ple` |
| `/home/jmicko/.codex/worktrees/kuno-target-grouped-status/vashti` | `codex/kuno-target-grouped-status` |
| `/home/jmicko/.codex/worktrees/kuno-target-only-controller/vashti` | `codex/kuno-target-only-controller` |
| `/home/jmicko/.codex/worktrees/kuno-text-ops/vashti` | `codex/kuno-count1-input-tail-fix` |
| `/home/jmicko/.codex/worktrees/kuno-three-gdn-parent/vashti` | `kuno-handoff-20261031/kuno-three-gdn-parent` |
| `/home/jmicko/.codex/worktrees/kuno-verify4-dense/vashti` | `codex/kuno-four-compact-expert-groups` |
| `/home/jmicko/.codex/worktrees/strata-branch-parameters/vashti` | `kuno-handoff-20261031/strata-branch-parameters` |
| `/home/jmicko/.codex/worktrees/strata-int8-kv/vashti` | `codex/strata-int8-kv` |
| `/home/jmicko/.codex/worktrees/strata-resident-first/vashti` | `codex/strata-native-cjk` |
| `/media/jmicko/faststore/kuno-development/review-experts-20261023/vashti` | `kuno-handoff-20261031/review-experts-20261023` |
| `/media/jmicko/faststore/kuno-development/review-fused-safety-20261023/vashti` | `kuno-handoff-20261031/review-fused-safety-20261023` |
| `/media/jmicko/faststore/kuno-development/review-hc-20261023/vashti` | `kuno-handoff-20261031/review-hc-20261023` |
| `/media/jmicko/faststore/kuno-development/review-pipeline-timing-20261023/vashti` | `codex/strata-public-pipeline-timing` |
| `/media/jmicko/faststore/kuno-development/review-quant-half-native-20261023/vashti` | `kuno-handoff-20261031/review-quant-half-native-20261023` |
| `/media/jmicko/faststore/kuno-development/review-quant-half-probe-20261023/vashti` | `kuno-handoff-20261031/review-quant-half-probe-20261023` |
| `/media/jmicko/faststore/kuno-development/review-quant-stage-slim-20261023/vashti` | `codex/strata-quant-stage-slim` |
| `/media/jmicko/faststore/kuno-development/review-small-quant-cohorts-20261023/vashti` | `codex/strata-small-quant-cohorts` |
| `/media/jmicko/faststore/kuno-development/review-small-quant-kernels-20261023/vashti` | `codex/strata-small-quant-kernels` |
| `/media/jmicko/faststore/kuno-development/review-small-quant-runtime-20261023/vashti` | `codex/strata-small-quant-runtime` |

## Pruned already-missing registrations

These checkout directories were already absent. Only stale worktree registrations were pruned; their branches remain.

| Missing checkout | Preserved branch |
| --- | --- |
| `/tmp/kuno-attention-benchmark` | `codex/kuno-attention-benchmark` |
| `/tmp/kuno-attention-layout` | `codex/kuno-attention-layout` |
| `/tmp/kuno-im2col-large-audit` | `codex/kuno-im2col-large-audit` |
| `/tmp/kuno-int8-profiler` | `codex/kuno-int8-profiler` |
| `/tmp/kuno-int8-tiles` | `codex/kuno-int8-tiles` |
| `/tmp/kuno-intel-flash` | `codex/kuno-intel-flash` |
| `/tmp/kuno-intel-query-tiling` | `codex/kuno-intel-query-tiling` |
| `/tmp/kuno-managed-runtime-fixes` | `codex/kuno-managed-runtime-fixes` |
| `/tmp/kuno-managed-sycl` | `codex/kuno-managed-sycl` |
| `/tmp/kuno-mlp-f16` | `codex/kuno-mlp-f16` |
| `/tmp/kuno-sycl-convrot` | `codex/kuno-sycl-convrot` |
| `/tmp/kuno-sycl-copy-offset` | `codex/kuno-sycl-copy-offset` |
| `/tmp/kuno-sycl-sdpa` | `codex/kuno-sycl-sdpa` |
| `/tmp/kuno-sycl-vae-metadata` | `codex/kuno-sycl-vae-metadata` |
| `/tmp/kuno-sycl-vae-review` | `codex/kuno-sycl-vae-review` |
| `/tmp/kuno-vulkan-cache` | `codex/kuno-vulkan-cache` |
| `/tmp/kuno-vulkan-cpu-params` | `codex/kuno-vulkan-cpu-params` |

## Runtime and artifact boundary

No GPU test, installation, service restart, driver change, reset or reboot was performed during cleanup.
Kuno, Ollama and Vashti HTTP health responded successfully, admission was inactive/unpaused with no recovery latch,
and no native benchmark unit was running. Normal managed inference workers were left alone.
Frozen release directories, external runners and prepared snapshots, public CODE caches, safe receipts and failures remain under
`/media/jmicko/faststore/kuno-development`. They are not disposable checkout clutter.
The machine-readable cleanup record is `kuno-handoff-worktree-cleanup-20261031.json` in that directory.
