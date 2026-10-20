# Connected capture and attention candidate

Kuno-only continuation, 2026-10-20. Installed worker3fffc266 / SHAeac78ce0
and the complete original model/settings stay selected. No HTTP, private-wire,
configuration, model identity, Vashti, release or installation change yet.

## Evidence and exact change

The complete native routed-wide/bulk/loading/host-batching pipeline6e61895b
passes129 numerical GPU cases but remains slower than installed native32.
Its protected public diagnostic samples3.7085s user CPU in the repeated4245
prefix, with distributed allocation/zeroing/planning/Wgpu lookup costs. Current
descriptor allocation/update calls are ALREADY batched per128 original nodes;
this change does not claim to invent or duplicate that batching.

Retain every complete pipeline component, then connect:

- An explicit `strata-qsa-local-fault` / builder `--qsa-local-fault` selection.
  Only `PromptAttentionFault` and `PromptMergeFault` lose fault-only shared
  reduction: every original causal/finite predicate sets an invocation-local
  sticky flag, published by atomic OR into the SAME real global fault owner.
  Attention keeps the three query/score/probability barriers, all chronological
  arithmetic and stores; merge keeps ordered stats/exp/FMA/divide and per-lane
  zero-on-invalid output. Request acceptance still reads the fault after the
  genuine fence. Ordinary status-producing variants, scalar numerical oracle,
  decode/MTP and all original numerical tolerances remain unchanged.
- A callback traverses the exact original bindings, physical read dependencies
  and indirect access without constructing another temporary reference Vec.
  Per-capture ranges, merged buffer access flags and buffer barriers reuse
  checked-capacity scratch, instead of allocating/dropping them for each node.
  The original Planner receives borrowed ordered ranges but retains its exact
  complete hazard/segment plan. Same-buffer aliases are OR-merged before the
  original previous-access map is updated. No raw-handle/destruction-guard
  cache or changed Wgpu access check is introduced.
- Recording lifetime resources retain one genuine clone per distinct original
  Wgpu buffer. This vector is used ONLY for buffer lifetime retention; original
  node views, full replay signatures, planner ranges, descriptor views and
  actual allocator Binding Arc bundles remain complete and unchanged. Every
  Recording independently owns its required roots through the actual fence or
  original unknown-completion quarantine. Equal content/different buffer
  identities never deduplicate.

Scratch bounds are the original32 descriptor +4096 physical dependency +one
indirect maximum, with actual Vec capacities and conservatively bounded hash
buckets charged below1MiB and included in the unchanged64MiB Graph allowance.
The temporary sorted borrowed resource index and its simultaneous actual
retained clone vector are checked against remaining original metadata budget.
The original parent reservation is not raised. No source, range, release event,
queue barrier, timeline wait, ownership/admission check or erasure is omitted.

## Validation and remaining gates

The first combined CPU QSA run passes39 controls, including unchanged original
bitwise/F64 oracle controls and actual Naga/SPIR-V checks for the transformed
sources. An earlier test run passed38 and failed the historical assertion that
PromptMergeFault was byte-for-byte original; that assertion's scope now excludes
the deliberately selected fault-only candidate. Non-fault paths remain exact.
The fault transformation's complete attention arithmetic is reversible and
merge's entire arithmetic prefix compares unchanged. All single-lane,
multi-lane, all-good and pre-existing fault-bit aggregation retain sticky OR.

Frozen complete worker sourcefa2138fc926c1c3b99ac4add5edc31814e8f489f,
SHA256 `802e3fa2b85c0d30037c9c636d5a0db599f9f78047b3c44cc91286971f1a50e5`,
build `strata-parity-capture-attention-20261020`, selects all previous ten
features PLUS the explicit local-fault feature. Final actual App compilation
and82 focused CPU controls pass:39 QSA,37 queue/planning/submit/recovery policy,
three binding/resource lifetime, one actual scratch reuse/bound and two bulk
serializer/budget cases. Release build and402 ordinary/208 matrix offline
layouts pass. Exact logs are `native-qsa-host-scratch-final-cpu-20261020.log`,
`strata-parity-capture-attention-build-20261020.log` and
`strata-parity-capture-attention-offline-20261020.log` under the artifact base.

All129 actual original GPU controls and the complete three-request profile now
pass, including real six-logical expert/wide-tail counts and physical recovery.
Receipts are `strata-parity-capture-attention-20261020-{operators,profile}`.
Load78.7574s; first/repeated4245 prefixes12.8209/12.0604s. Repeated expert wall
7.7714s, capture2.5667s, enqueue1.5450s, wait3.4883s; QSA attention/merge1.46846s.
The modest improvement over the complete predecessor12.3522s is NOT a clean A/B
and remains behind installed11.1474s and same-host Strata5.3515s. All48,851
original segments/1,474 rolling windows/39,225 HOST copies remain. The host-after
receipt confirms Intel50,688,000B, NVIDIA compute0, no new driver faults,
Kuno/Ollama/Vashti200 and no active/waiting/paused/recovery admission.
This artifact is unselected and uninstalled; full staged/terminal acceptance
and a new sustained comparison are not claimed.

## Connected original-view transfer dependency candidate

The next candidate retains the whole fa2138fc pipeline and changes ONLY
same-COPY-queue memory barriers. Cross-queue timeline dependencies, original
command segments/order, full source/destination views, actual INFO/binding
owners, fault/numerical checks and all release points remain unchanged.

Previously one whole-buffer access flag per queue inserts a barrier between
adjacent disjoint role copies. Merely using exact intervals is insufficient:
raw-ring slot reuse would still encounter earlier writes. The coherent candidate
therefore connects BOTH exact original-view intervals AND actual existing
ALL_COMMANDS peer-wait ancestry from the completed original queue Plan.

A pure bounded visibility builder validates every original segment signal,
queue, backward peer wait and header; its three-queue frontier derives which
COPY epochs were already acquired by a genuine incoming peer semaphore wait.
Same-queue order alone grants no memory synchronization. An old interval may
omit a redundant barrier only when disjoint or when its COPY epoch is covered
by that actual acquire. A later write in the same segment is never covered.
For every remaining overlap, the original TRANSFER stage and whole-buffer
barrier/access masks remain. Original full descriptor views are never narrowed
to the smaller physical copy byte count. Within-node aliases are canonicalized
before update so no fake intra-command dependency or forgotten read is added.

All transfer dependency arrays are built before the first GPU prefix. Their
real capacities, temporary frontier/interval/hash capacities, original live
Plan and allocator owners, plus the simultaneous borrowed resource index and
retained resource clones, are preflighted under the existing64MiB Graph cap.
The original parent reservation is not raised. The public-only profiler adds
encoded COPY-node/barrier aggregate deltas after the original real sync; there
are no prompt identifiers, text, route traces or persisted private metrics.

Four initial CPU controls pass: an8192-node independent byte-addressed hazard
oracle (aliases/gaps/reads/writes), disjoint copies/real slot reuse, genuine
three-queue wait ancestry/same-segment negatives, and invalid/bounded inventory.
The first compile caught a duplicate existing test-module alias; that fixture
integration is fixed. Final complete actual App compilation and86 CPU controls
pass (all previous82 plus those four transfer controls), in
`native-transfer-visibility-final-cpu-20261020.log`. GPU acceptance is pending. No new speed, terminal or install claim for this change.

A meaningful clean gain is needed before full staged tools/MAX8/semantic/privacy
and injected terminal recovery, then official installation and actual installed
service acceptance. The goal remains matching actual same-host Strata repeated
4245 prefix5.3515s and matched sustained inference.
