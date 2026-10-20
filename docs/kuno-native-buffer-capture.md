# Scoped native buffer capture

Kuno-only internal continuation, 2026-10-20. No network/protocol, model identity,
configuration, numerical, frontend or installation change. Keep the complete
routed-wide/bulk/miss-ring/attention/loading/transfer pipeline.

## Evidence and contract

The protected d95a12da DSO-aware leaf profile distinguishes genuine Rust/Wgpu
storage/registry/destruction-lock costs from Mesa leaves. Repeated4245 prompt
samples3.2513s user CPU, including0.1206s storage lookup,0.0704s registry lookup,
0.0503s snatch-lock read and other related allocation/access costs. These are
sampled leaves, not a promise that their removal saves the same wall time.
The complete bundle's clean11.2306s remains level with installed11.1474s and
behind Strata5.3515s. No single driver bottleneck is established.

Resolve original buffer roots once per capture phase under a real Wgpu guard:

- Collect the same complete genuine buffer identity inventory used by Recording
  lifetime retention, including every descriptor, full BDA dependency and
  indirect root. Equal bytes never establish buffer identity. Original per-node
  views, access order/flags/ranges, signature fields, INFO ownership, queue Plan,
  all admission checks and actual allocator Binding Arc bundles remain intact.
- Use pinned Wgpu/wgpu-core29.0.4 with a narrowly named Kuno batch HAL extension.
  Original `as_hal` remains unchanged. A batch first resolves/retains the actual
  core Arc owners and validates one exact device/context. It allocates checked
  bounded host containers BEFORE taking one device destruction read lock, then
  checks every live Vulkan raw resource under that lock. All fallible paths
  release the real lock; successful construction forgets exactly one guard.
  Drop unlocks before any retained resource can be dropped. No bare-handle
  cache is retained without its guard, and no nested read lock is acquired.
- The same complete sorted real buffer keys serve signature construction,
  planning/transfer visibility and Vulkan descriptor/copy/fill/indirect/barrier
  encoding. Binary lookup rejects an absent root; actual descriptor offsets,
  sizes and checks remain unchanged. Guarded HAL references borrow the batch.
- Guard phases are deliberately separated: release after signature construction
  BEFORE replay/eviction; after planning BEFORE original flight preparation can
  retire old owners; after encoding BEFORE submission bookkeeping, owner drops,
  Wgpu completion/polling, erasure or cleanup. Rolling prefix operations and
  recovery still use original Vulkan waits/fences. No guard crosses a request.

Actual unique lifetime resource storage, original complete signature, all
existing Recording/Binding/Plan metadata, capture key vector capacities, and
core guard vectors plus temporary BufferIds are checked inside the original
64MiB Graph allowance. Each capture phase reserves32B per root plus4096B for
guard/temporary headers, with actual allocation capacities rechecked by both
APIs. Roots remain bounded65536. No GPU allocation or admission increase.

The added vendored crates are the published MIT OR Apache-2.0 `wgpu29.0.4` and
`wgpu-core29.0.4`, copied from the local Cargo registry with original license
files. Their original package/source/README files remain; generated Cargo
cache markers and lockfiles are excluded. Changes are limited to the named
batch extension/export. Existing wgpu-hal remains its separate prior fork.
There is no new production dependency/version and no native C++ inference.

## Validation gates

The actual complete App test build compiles the connected code. Original101
App CPU controls are rerun; this is distinct from numerical GPU acceptance.
A new explicitly ignored, protected real-B70 lifecycle control verifies live
batch bounds, an actual competing destruction writer blocking while the guard
is held and completing after release, destroyed-resource rejection, mixed-device
rejection, and a fresh original single-resource lookup after failure. It accepts
no prompts or model data. Ordinary test discovery never runs it on a GPU.

Then require original129 numerical GPU controls, all408/208 offline layouts,
complete original-weight requests/actual producer counts and physical recovery.
Only a meaningful clean matched improvement justifies full original staged
thinking/tools/MAX8/semantic/privacy, terminal-prefix/fresh recovery, official
installation and actual installed service checks. A successful batch guard
control alone does not establish model correctness, throughput or safe install.

Actual App test compilation and101 original CPU controls pass, log
`native-capture-buffer-cpu-20261020.log`. The inherited empty
`release_metadata_tests` filter was rejected before execution; the actual two
`bulk_metadata_tests` run and pass, as do all other original suites.
The protected real B70 lifecycle control passes, executable SHA256
`05a77720d2c372ce43dda8d9b6aa08090ba167efb9cb9c4cd0f76b9783892c4a`,
receipt `native-capture-buffer-guard-lifecycle-20261020/receipt.json`.
Actual post-test memory/fault/admission/HTTP recovery passes. An initial test
fixture used an unsupported InstanceDescriptor Default; it was corrected to the
same explicit no-display constructor used by the worker before final testing.

The published source comparison confirms only two modified files in each newly
vendored package: wgpu API/backend wrappers and core guard/export. No other
upstream crate code is changed. Release/offline/full numerical/request gates
remain next. Current source is development only; no measured speed or
installation acceptance.
