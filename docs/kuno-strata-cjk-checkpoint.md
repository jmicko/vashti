# Native CJK draft-head source checkpoint

Status: connected source implementation, **uncompiled and unexecuted**. This is
isolated on `codex/strata-native-cjk`, based on integration commit `19aa565`.
Do not claim GPU correctness, performance, full rewrite completion, or release
readiness from this checkpoint. Parent explicitly stopped work before a new
integration/build loop. No GPU, service, sudo, build, or test commands ran here.
`rustfmt` and `git diff --check` passed. Existing main task changes remain intact.

## Implemented as one connected component

- `strata_mtp_cjk.rs`: original-order106299 public IDs, complete inverse map,
  separately owned gathered target Q6_K rows, exact public page plan, startup
  validation and retained full owner handles. Selection is mandatory for the
  native Intel core with StrataQ2, excluding the old Q8 English/indexed option.
- `resident_mtp.rs`: load/charge/validate/retain the complete CJK owner; project
  compact logits using the native Q6 path; select compact argmax and compute
  compact-domain confidence before mapping to original IDs. The full-logit
  Output API rejects this draft-only policy instead of fabricating omitted
  probabilities. Existing full target output/verification is unchanged.
- `strata_mtp_probability.rs` and `strata-mtp-cjk-map.wgsl`: native compact
  argmax/probability variants and final original-token mapping.
- `strata_sampler.rs` and `strata_sampling_request.rs`: coupled draft sampler
  uses106299 rows/26 split blocks. Device history uses the inverse map (absent
  original tokens map to-1); selected token is mapped back before publication
  to the original-ID ring and draft token. Target sampling stays248320/61.
- Original public fixture plus README provenance. The old sorted fixture is
  preserved for its original legacy consumer. Ordering matters for exact ties.

## Exact upstream source and ownership

MIT Strata revision `fb58e0dbc8399662c0e47c76578c6e878b14f6cf`, pinned checkout
`/home/jmicko/.codex/worktrees/kuno-native-split-gguf/strata-research-20261009`:
`sycl/src/core/mtp.cpp:749-778` gathers target-head rows without Q8 conversion;
`:631-644` constructs its inverse map; `:1123-1145` projects, samples, computes
compact confidence, then maps back to original IDs.

`data/draft_vocab.bin` SHA256:
`b1e1d3a7a9e4bf862dcd5923ce661fb59bbd07907e594df5cf86a62ac235cb91`.
The exact original bytes are now the original-order fixture. Loader copies
original GGUF Q6_K row bytes directly into private-to-the-owner public upload
buffers; there is no dequantization, requantization, persisted weight cache,
or request-content storage. Gathering is performed once in Rust from source
GGUF rows, rather than via the upstream GPU gather kernel.

Packed head has223227900 logical bytes,223227904 owned bytes in seven pages
(each at most16384 rows,4096-aligned); IDs524288 bytes and inverse1048576 bytes.
Total new public device charge:224800768 bytes. Existing full target shared
head remains retained. Admission derives shared capacity from public_plan_for,
so this charge is added once. Existing private logits allocation remains full
size, but only the real compact prefix participates in draft sampling. All
existing private arena scrub/retirement paths remain active.

## Deferred validation and integration

1. Cherry-pick this source checkpoint only when ready for a new build. Other
   agent changes in `resident_mtp.rs` are distinct: draft INT8 KV Append dispatch
   and target semantic capture/role buffers. Resolve by retaining both.
2. Run CPU compile/check on the parent feature set (`strata-intel-core`), then
   relevant unit filters: CJK fixture/order/page plan, native compact confidence,
   native sampler mapping, and native bridge shader lowering. All five native
   confidence variants, six sampler variants, and four bridge variants are
   included in the existing Naga/SPIR-V offline source checks. Tests were added
   but not executed here; no passing result is implied.
3. Exercise protected whole-model startup and actual MTP greedy/coupled paths.
   Verify new public-byte ledger and source readback budgets, original-ID ring,
   compact confidence, sticky faults, complete owner cleanup and capture reuse.
4. Compare actual greedy/coupled draft decisions with pinned Strata under the
   same source, subset order and settings; full target verification must retain
   every original vocabulary token. This remains hardware acceptance pending.
5. Exact sampler head source variants use existing native shader arithmetic;
   this checkpoint does not resolve unrelated already-recorded whole-engine
   completeness gaps. Do not optimize away features based on a speed result.

No API/frontend changes or installed-service changes are part of this work.
