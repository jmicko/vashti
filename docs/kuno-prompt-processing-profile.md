# Native prompt processing investigation — 2026-10-19

This is Kuno backend work in `codex/strata-rust-port`. No Vashti source or API
change is involved. The installed toolbox repair remains the deployment baseline
until the acceptance/install section below explicitly records a replacement.

## Workload and measurement

The dedicated `--public-prefill-{profile,check}` entry accepts a model-settings
path, never stdin or caller text. It runs three compiled public requests on one
Session: 45 input tokens, then the same 4,245-token input twice. The longer
input adds 300 repetitions of public reference material before the same arithmetic
question (17 times 19). Temperature0, seed0, context32768, output cap32; each
completed request returns four tokens containing323. Reply bytes remain in RAM.
Actual request erasure and Session close are required before a passed receipt.

`--public-prefill-profile` adds completed GPU fences at coarse operator boundaries.
Its times are diagnostic, NOT ordinary throughput. The static per-category tuple
is `(calls, elapsed_ns, enqueue_elapsed_ns)`; the first baseline predates the
third field. Enqueue time includes any blocking inside the called backend, not
just CPU arithmetic. Nested attention categories overlap their outer category
and must not be summed twice. `expert_cpu_*` slices do not add device fences.
The private worker cannot activate any profiling through input or environment.

The reported prompt interval ends before the final count1 input, as in the
installed worker. Loading, generated output and completed-request erasure are
separate. Neither prompt-cache retention nor the privacy contract is changed.

Receipts/artifacts below are under `/media/jmicko/faststore/kuno-development`.
Every protected runner disables swap/core dumps, unloads the native worker,
pauses gateway admission, uses the real B70, then requires graceful close,
restored physical memory, normal admission and unchanged driver-fault counts.
Only static/public metadata and numeric results are saved.

## Measured baseline

- Source `342970e2`, artifact `strata-prefill-profile-baseline-20261019`, SHA
  `62c9f917eb8b75c8a52ab8fc4dc21a27faa7942358ef8e10c8ed0209d79ca5ff`.
  `prefill-profile-baseline-20261019/receipt.json`: warm repeated prompt56.596s,
  experts35.532s, attention15.934s, MTP0.187s. All three replies/correctness,
  request erasure/close and physical recovery pass.
- Source `4a33422e`, artifact `strata-prefill-member-grids-20261019`, SHA
  `df9a77bffbd1604975a0a9999d394f16db029ab37e7408b5bed1a0e49c40bc94`.
  `prefill-member-profile-20261019/receipt.json`: warm54.998s, experts33.966s,
  attention15.926s. Bounding gather/activation grids to the actual expert's
  routed members saves about1.6s in this diagnostic, but is not the main fix.
- Source `f9471ce8`, artifact `strata-prefill-attention-profile-20261019`, SHA
  `6388885a8bd7b6007b08d00911330ea593fed7b3782e30a556830f9369974cfb`.
  `prefill-attention-profile-20261019/receipt.json`: warm55.231s. GDN input
  projections6.637s/output2.543s; QSA input1.958s/output0.852s; QSA attention
  3.006s. GDN recurrence/norm only0.892s, so register-state/recurrence changes
  are not selected. This diagnostic also passes complete recovery.

## Selected changes and validation scope

- `342970e2`: compiled-public profile/check callers and fenced coarse timing.
- `4a33422e`: expert member-bound gather/activation grids, keeping the full
  original descriptor and shader ownership bounds; adds grid coverage checks.
- `f9471ce8`: attention projection/recurrence subdivisions, diagnostic only.
- `ef886df5`: queue original same-stream expert kernels/copies without an
  intermediate scheduler flush per expert. The native graph retains original
  physical read/write intervals, ring release dependencies, quotas, error scopes
  and final actual completion/fault checks. Ordinary non-native scheduling still
  flushes. Adds non-submitting original-order scheduler tests. Also stages K64
  shared matrix operands, issuing the same four ordered K16 cooperative products
  between barriers. FP16/BF16 rounding and accumulation type remain unchanged.
  The dedicated compiled-public matrix check uses the existing independent IEEE
  oracle, original model weights, full-owner/tail checks and erasure audit for
  counts1/8/35/257 in both formats (eight cases); no looser FTZ oracle is adopted.

CPU status at candidate preparation:27 prompt tests and strict selected worker
Clippy pass. Three scheduler tests pass by compiling the ORIGINAL vendored
scheduler source with a tiny Rust harness and the lab's actual dependencies.
Direct `cargo test -p cubecl-runtime` cannot run non-workspace dependency dev tests;
no vendor manifest/workspace membership was changed to work around that limit.
Harness: `prefill-scheduler-test-20261019.rs`; log
`prefill-scheduler-direct-tests-20261019.log`.

## Acceptance and installation

Pending physical numerical, whole-model, uninstrumented before/after, private
API/tool/streaming and recovery validation. No new installed speed claim yet.
