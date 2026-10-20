# Strata short prompt path and bounded Rust implementation — 2026-10-19

## Source finding

Pinned upstream is MIT Strata `v0.1.41`,
`fb58e0dbc8399662c0e47c76578c6e878b14f6cf`, inspected locally at
`/home/jmicko/.codex/worktrees/kuno-native-split-gguf/strata-research-20261009`.
These are selected Intel/SYCL sources, not inactive CUDA/HIP alternatives.

**The historical short API timings are not measurements of the same prefill
GEMM path as Kuno's35-token prompt.** Upstream serving defaults to a separate
short-read strategy:

- `sycl/src/program/generate.cpp:664–667`: `short_read=64`, counting each fresh
  text segment's inputs. `:808–809` documents `--short-read0` as disabling it.
- `:9720–9734`: segments of at most64 tokens use verification windows; image
  embedding rows and `STRATA_CKPT_REREAD` require the batched path. Selection is
  per segment, so a long message and its short assistant header can differ.
- `:10189–10203`: selects `read_windows` or `read_part`; the latter calls the
  normal batched `Prefill::run` at`:10045`.
- `:9853–9898`: `read_windows` uses the existing verifier in `S=o.spec` chunks,
  commits all known inputs, disables head sampling, and supplies actual residuals
  and known next tokens to MTP. It does not produce FP16 images of each expert.
- `sycl/CMakeLists.txt:211–217` builds the Intel prefill with oneMKL and the
  fused-kernel stub. `sycl/src/prefill/prefill.cpp:53–64` makes MMQ unavailable;
  `:842–850` therefore disables its plan. When the batched path is selected,
  `:3964–3983` dequantizes the whole original expert GU/DOWN into two FP16
  images and calls two GEMMs; `sycl/src/prefill/gemm.dp.cpp:799–805` dispatches
  those products to oneMKL. No additional short-token fused threshold exists in
  this selected batched expert flow.

Historical public API receipts are
`/media/jmicko/faststore/kuno-development/strata-comparison-evidence-20261009/receipts/public-api-check.json`:
26/35/30 tokens take0.7873/0.7058/0.6373s with zero cached tokens. The archived
`manage.py.txt:58–73` starts the normal Intel server from `strata-iq3_s.json`,
with a cleared environment and no explicit short-read override. The original
JSON engine-argument file is not present in that retained evidence directory;
the Python server can pass arbitrary `cfg["args"]` (`serve/server.py:2165–2169`).
Thus the selected default and applicable threshold are source-proven, but the
historical receipt alone does not prove which exact engine branch ran. Do not
label those numbers a controlled comparison of the35-token FP16/GEMM path.

## This Rust checkpoint

An internal branch selects the existing quantized native model path for a fresh
prefix of at most64 inputs, excluding the final ordinary count1 prompt input.
It uses the existing `prefill_chunk_with_carry` with at most THREE rows, not the
complete upstream MAX8/no-head/device-carry implementation.

For each window the controller checks the actual target and draft entry cursors,
executes and fully commits the original target chunk, verifies its opaque result
coordinates and completed target cursor, copies actual fenced residual rows into
a bounded `Zeroizing<Vec<f32>>`, drops the original output-host lease, and calls
the existing MTP `append_kv_group` with the exact next INPUT tokens. Both completed
cursors must then equal the window end. Greedy predictions computed by this path
are ignored; they never replace known teacher tokens. Without MTP, the same
target completion checks apply and no host residual copy is made.

The maximum extra wiping RAM carry is3×10,240×4=122,880bytes, included in the
existing4MiB request HOST bound. It does not add a GPU owner, fabricated carry
grant, prompt persistence, wire field or setting. Longer
prefixes retain the existing layer-major prompt path/cache loan unchanged. The
final prompt input, generation policy selection, sampler history and original
request erasure/rebind/close path remain unchanged.

An initial protected run correctly rejected the legacy full-commit entry with
`native submission requires original request reuse`. The coupled correction
adds a distinct **ShortPrefix** authority with an actual original runtime nonce:
only cursor0 entry,1..64 known prefix inputs, MAX3 per window, and no generation
scope credit. Entry checks the original model/erasure boundary before beginning
the actual nonce. Closing checks no live output lease, completed native families,
the original fault word and the real full owned I/O closing audit before returning
to Prefill. The original final count1 InitialWindow and long generation scope
remain separate. No backend guard is weakened or caller-supplied nonce invented.

Warm cursor0 checks retain the original `RetainedReady` provenance through a
dedicated short-scope check: same fault/binding/context, erased PLE/history state,
matching previous erased scope and unchanged generation counter. The original
no-nonce retained checks are unchanged; the used PLE generation is never replaced
with a fresh constructor stamp. Actual short-prefix cursor checks use that bounded
authority while its runtime nonce is live.

The tradeoffs are real: MAX3
windows still compute their original heads and read compact residuals to wiping
RAM before teaching MTP. It is a functional stepping stone, not claimed full
Strata short-read parity. Protected actual short-request acceptance is recorded
below; complete mixed API acceptance remains separate.

## Remaining complete path

The language I/O authority now distinguishes Prefill (at most3rows without
the generation nonce), bounded ShortPrefix with an actual audited nonce,
InitialWindow (exact final count1 input), and generation.
An existing `DeviceCarry` also authenticates actual predicted tokens; it cannot
be reused for arbitrary known next teacher tokens. `append_prompt_device` needs
the large prompt scope and its original workspace. A full MAX8/no-head/device
short-read needs a connected implementation, not a relaxed prediction check:

1. Extend the dedicated bounded known-prefix I/O phase from MAX3 to the genuinely
   reserved MAX8 policy after the real original unstarted/retained boundary.
   Keep final-input and generation admission distinct; do not fabricate an
   accepted-prefill stamp.
2. Add a known-input whole-commit purpose that omits the unused final head and
   exposes actual fenced residual rows using its own non-prediction capability.
3. Teach the exact next input IDs to MTP from that genuine device residual source
   under an actual bounded visibility scope. Join real MTP/fault completion before
   releasing the source; verify both cursors and preserve original target state,
   PLE history, QSA cache and recurrent commits.
4. Preserve the final count1 handoff and existing complete request erasure,
   retained model rebind, cleanup/failure and protected recovery gates.

## Validation and next protected acceptance

The actual pure planner is compiled directly with the existing dependency rlibs:
two CPU tests pass, covering empty/final-only and1/3/4/64/65 input boundaries,
every shifted teacher slice, tails, invalid lengths/ranges and overflow. Evidence:
`/media/jmicko/faststore/kuno-development/prefill-short-prompt-plan-tests-20261019.log`.
Four further CPU cases compile the actual pure Geometry/Authority source and its
actual tests using existing dependency rlibs, with selected-profile constants:
two previous InitialWindow cases and two new short-scope/rearm cases pass. They
cover true/false actual-nonce state, MAX3 even with MAX8 reserved, threshold and
cursor bounds, no early generation/rearm/closed-scope credit, matching original
erased scope, and the unchanged final count1 handoff. Evidence:
`/media/jmicko/faststore/kuno-development/prefill-short-prefix-authority-tests-20261019.log`.
These are metadata lifecycle tests, not backend nonce or model completion proof.
Rustfmt and diff checks pass. Those isolated CPU checks alone do not prove a
backend nonce or model completion. The integrated source7962fe68 now builds as
`strata-prefill-short-scope-20261019`, worker SHA256
`76d5f706e17f741901652121d193312ceda9955616f912bdf0810f048b5339be`.
It passes all14 matrix and39 actual router/dequant GPU checks with unchanged
references and real close/recovery. The compiled-public short check completes
three45-token arithmetic requests correctly in one model Session: cold prefix
10.0511s, two warm prefixes1.8118/1.7496s, zero repeated load time. Each returns
4tokens; original per-request erasure and retained rebind, final close and actual
physical recovery all pass. Model startup remains104.5667s. Evidence under
`/media/jmicko/faststore/kuno-development/`:
`prefill-short-scope-operators-numeric-20261019/receipt.json` and
`prefill-short-scope-warm-20261019/receipt.json`.

Protected GPU acceptance must include short requests around the actual threshold
(prefix63/64/65), repeated requests on one loaded model and short↔long transitions;
correct known target/MTP cursors; public arithmetic, code/prose, thinking and real
tool-call/result flows; zero worker disk writes; actual request erasure, unload,
driver-fault and physical memory recovery. Compare matched public prompt timing
only after correctness passes. Quantized decode math can legitimately change
output from the old FP16 prompt path, so a changed362-token code fixture requires
explicit output/quality review, not silently weakening an exact-output assertion.
No new live Vashti acceptance or installed/release claim is made here.
