# Strata short prompt path and bounded Rust implementation — 2026-10-19

## Prepared whole-request terminal-count coverage — 2026-10-20

`apps/kuno/benchmarks/text/public_short_tail_cases.py` is a fixed-public helper
for the protected staged acceptance client. It makes only ordinary `/api/chat`
requests through that client's existing `doc` callback and verifies its actual
selected-worker PID with the client's `pid` callback. Call
`run_public_short_tail_cases(doc, pid, first)` after preloading the native worker.
There is no worker, inference, API, policy, capture or authority modification.

The first fixed arithmetic prompt is calibrated to exactly50 total input tokens
in at most three requests. Single-word filler increments must then produce exact
totals50..57. Excluding the final ordinary count1 input gives prefixes49..56:
six full MAX8 windows followed by each terminal count1..8. The helper repeats
57..50 in reverse on the same loaded worker to exercise retained rebind and
changed capture shapes. Every request must finish normally below the32-output
cap, return exactly323, retain the original PID and report zero load duration.
Failure to obtain adjacent exact token counts rejects the fixture; no token
array, prompt, reply or content hash is returned or logged.

Only numeric request metadata is returned. Its explicit
`request_checks_passed_before_session_close` flag is not whole-session acceptance:
the parent staged client still has to check actual erasure/close, worker writes,
driver events and physical GPU/admission recovery. Existing total64/65/66
threshold cases, tools and semantic-code checks remain separate and unchanged.
Four CPU protocol/schedule tests pass, including bounded calibration, every tail,
reverse reuse, wrong arithmetic, truncation, reload/replaced PID and token-count
rejection. No GPU or live API execution is claimed by this preparation.

## MAX8 projection-authority repair — 2026-10-20

The first protected MAX8/no-head/device-carry request correctly rejected an
uncoupled projection policy before publication. A compiled-public-only static
diagnostic identified `native Verify3 projection count`: the known-prefix
controller reserved MAX8 and held its real ShortPrefix nonce, but its executor
still authorized only3-row projections until generation enabled a distinct
four/eight-row policy. Toggling generation early would also change GDN/PLE
selection and credit the wrong scope; that is not the repair.

The connected repair mints an opaque, single-window `ProjectionWindow` only
through the original ShortPrefix permit after exact graph/context/cursor/count
and live runtime-scope validation. The executor ties this window to the original
admission ledger, graph binding, and actual nonzero request nonce. Each dense
projection checks the same nonce and exact window count before any weight/cache
mutation. Counts4..8 select the existing native Verify8 shader through its
original Verify4 wrapper, preserving its original packed-weight decoding,
full owning input/output spans, finite/fault checks, and capture identity.
Counts1..3 retain the original Verify3 path. No shader arithmetic changes.

The projection permission exists only during that complete target invocation.
It restores on normal return, error, and unwind; failed or replaced-nonce
completion retires the original admission without clearing/refunding the actual
nonce. The model's real closing audit still owns nonce erasure. Generation's
four-row flag, GDN/PLE selection and Verify4 completion counters are unchanged;
known-prefix launches cannot credit those counters. The existing headless
recording key, genuine final residual receipt, exact teacher tokens and original
draft visibility/fence remain in force.

Six pure CPU checks pass using original lifecycle/geometry/nonce helper bodies,
including all counts1..8, final-input exclusion, prefix64/65, wrong phase,
missing/replaced nonce and original generation-scope accounting. These tests do
not fabricate a graph binding, GPU owner, admission, or backend nonce proof.
Evidence: `prefill-known-projection-scope-pure-tests-20261020.log` under the
development artifact base. Full worker type checking and actual cold/warm,
count/tail/capture, requesterasure and physical-recovery acceptance remain
parent-owned gates; this source checkpoint is not an installed speed result.

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
The final selected sourcec8db19de keeps this short implementation and restores
the faster complete6dfaf5ef shared-buffer matrix. Its exact staged API now passes
total64/65/66 inputs (prefix63/64/65), same-worker mixed long/short transitions,
correct arithmetic and two identical complete35-token code requests. Warm code
prefix1.3063/1.2901s. Quantized-prefix answers legitimately differ from the old
FP16-prefill answer:501tokens rather than362. The old384-output cap failed before
completion; the separate quality fixture permits1024, requires EOS, identical
repeats and155 semantic merge cases per answer in a strict unprivileged child.
No generated code executes as root or persists, and no math oracle is relaxed.
Ten-tool thinking/call/result, actual erasure/rebind, worker write_bytes0, denied
inspection, unload, subsequent Ollama and full physical recovery also pass.
Receipt: `native-prefill-selected-short-stage-quality-check-20261019/receipt.json`
under the same artifact base above. Official installation preserves the original
gateway/configuration; final installed requests, semantic repeat and deployment
metadata pass too. Warm installed35-token prefix1.4305s,501 outputs/15.2196s,
actual nonroot/model mount, swap/core0, zero worker writes/OOM/driver faults and
healthy services/admission. Weights remain warm; all test clients/staged services
and the task sudo refresh loop are stopped. See the
[service acceptance](kuno-native-text-service.md#native-strata-prompt-optimization-2026-10-20).
No fresh live Vashti or release claim is made here.

## Connected MAX8/headless/device-teacher candidate — 2026-10-20

The installed `c8db19de` / `fb31e124` checkpoint above remains the measured MAX3
implementation. This successor is a **CPU-checked candidate**, not a new GPU,
installation or performance result. It implements the previously missing coupled
path together rather than deciding on any individual change's speed:

- Fresh prefixes through64 inputs use the genuinely reserved MAX8 quantized
  model window. `ShortPrefix` still owns the actual runtime nonce, excludes the
  final ordinary input, and receives no generation-scope credit. The existing
  unstarted/retained-erasure checks and full closing I/O audit remain in place.
- A distinct `KnownPrefix` purpose executes the original target layer math and
  all whole-model fault, PLE, GDN, QSA and MoE acceptance/commit proofs, but runs no
  vocabulary head or target-pick/output packet. It creates a source receipt only
  after the actual full commit. Legacy prefill and verification cannot request
  or consume this distinct output purpose.
- The receipt identifies the original residual `BufferId`, opaque graph binding,
  actual row-generation and fully committed window. An immutable model/arena
  borrow authenticates those identities again before another model can read R.
  No predicted token, synthetic hidden vector or caller-declared pointer/range
  is substituted. Exact next INPUT IDs come from the original request slice;
  overlapping teacher/input rows must match, and the MTP consumer checks the
  exact immutable teacher slice and start/count again.
- The real original MAIN→draft visibility scope pins that R owner while the
  existing native MTP front copies its bounded R rows device-to-device, computes
  the original front math and joins its actual full fault/ledger fence. Only then
  can the source borrow report completion and the visibility scope close. MTP
  advances its KV cursor without issuing a prediction/full-forward stamp. An
  abandoned in-flight copy or visibility scope retains the original retirement
  behavior. There is no host residual allocation or download/re-upload.
- Headless and head-producing target programs have a separate immutable
  `NativeStrataBranch` inventory flag. Its family1-only setter changes recording
  identity, not input, nonce, submission, commit or erasure authority. All
  existing constructors default to the original head-producing inventory.
  This prevents a retained target graph from accidentally restoring omitted
  head work or replaying a headless graph for generation.

The final count1 input, real generation sampling/policy, prompt history,
request erasure, retained model rebind, long4096-token path, API and Vashti
integration are unchanged. No transfer-runtime or matrix implementation changes
are part of this candidate. Known-input metadata uses the existing8KiB carry
quote and existing MAX8 owners; the full-worker test includes its actual type
size check. The earlier MAX3 host-carry allowance in the controller is removed.

Nine focused CPU tests pass by compiling actual pure source bodies and tests
against existing dependency rlibs, without a shared Cargo build. They cover
MAX8 tails and exact teacher slices,64/65 threshold, real-phase nonce/cursor
requirements, no generation credit/rearm bypass, source-generation/full-commit
identity, changed teacher rejection, and distinct headless/head recording keys.
The standalone extraction uses generic numeric binding IDs and public geometry
constants, and changes only helper visibility to fit its local module. It is not
an original GPU-owner, backend-nonce or device-completion proof. Evidence:
`/media/jmicko/faststore/kuno-development/prefill-known-device-pure-tests-20261020.log`.
The original actual-type quote assertion remains in the full crate tests.

Next acceptance requires the complete selected-worker build, cold and repeated
short requests on one model, prefix63/64/65 boundaries, short↔long transitions,
correct arithmetic/code/prose, thinking/tool result use, and all original
privacy/erasure/unload/physical-recovery checks. Numerical references and
semantic-quality acceptance must remain unchanged; a changed complete response
alone is not a regression or proof of quality. Measure matched short/long prompt
and generation performance only after those complete checks pass.
