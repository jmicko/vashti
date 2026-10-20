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

Final combined source compilation, scratch/owner/planner controls, full offline
layouts,129 original GPU controls, complete real requests/counts and recovery
remain required before performance selection. A meaningful clean gain is needed
before full staged tools/MAX8/semantic/privacy and injected terminal recovery,
then official installation and actual installed service acceptance. Do not call
CPU compilation proof of runtime recovery or compare diagnostic sampled-layer
timestamps as full-request throughput. The goal remains matching actual same-host
Strata repeated4245 prefix5.3515s and matched sustained inference.
