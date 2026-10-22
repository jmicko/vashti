# Authentic cohort recording selection

Internal Kuno experiment, based on31f6e54b. No API/model/configuration/frontend
change. This supplements the complete small-expert cohort implementation; it
does not select that implementation or establish GPU correctness/performance.

Batching removes real dispatches. Consequently the original rolling/idle-pool
minimum1024 compute nodes and diagnostic minimum2048 can exclude a substantial
cohort workload. The original selectors and thresholds remain unchanged for
every workload without independently authenticated cohort evidence.

The backend calls `Stream::admit_native_expert_cohorts(&scope, &grammar)` only
after the original complete grammar has passed. Admission itself calls the
genuine grammar's `finish` again, validates the original live private I/O scope,
and binds the actual group/phase/launch inventory to that scope's original
owner, epoch, generation, reuse identity and current Graph node offset. Neither
kernel names, public flags nor supplied numeric counters can mint admission.
The original successful batch close seals the exact count of genuinely queued
compute nodes. Failures still finish the original I/O scope and retire it.

Automatic submission is deferred only while that bounded authenticated batch
is being enqueued; explicit submission of an incomplete batch fails closed.
At capture a new temporary witness checks the sealed range against the actual
current Graph identity and node/compute population. Original append-only
commands may follow the batch, but their actual complete counts are bound too.
The witness allows the small cohort population to use the existing MAIN,
unnamed, unprofiled rolling path and completed-pool preparation. All original
physical roots, source/owner/INFO/epoch checks, replay signature, queue hazards,
protected draft checks, metadata caps and final joined flight remain required.

Idle retirement retains a separate opaque admission bound to the original
recording ID, family, Graph and actual node/compute population. It extends only
the old size predicate; live-flight, ready, unnamed, push-only, nonprofiled,
correct-family, reset and metadata admission remain unchanged. Pool objects
retain no prior request proof. Re-encoding derives new admission from the new
genuine batch; replay cannot create admission for another invocation.

The same temporary witness extends only the diagnostic size predicate. The
existing compiled-public sampler authorization, actual timestamp support,
16384-node/query bounds and original profiling/serialization semantics remain.
Counters always report genuine launches, nodes and logical phases separately.

The proof uses fixed-size metadata, explicitly charged to the existing capture
and recording caps; no new allocation or cap increase. Scope identities and
numeric proof fields are erased on destruction. Pending/closed evidence clears
at actual submission, and no proof/log/content is persisted.

Six focused CPU controls cover empty/incomplete grammar, exact actual scoped
population, foreign/stale owner and generations, changed Graph/node/range
counts, recording ID/family binding, forbidden queue/named/profile selection,
failure cleanup, bounded metadata and in-place erasure. Original threshold
controls remain unchanged. At this source checkpoint the new module was parsed
with rustfmt and diff checking passed; CPU compile/test execution is delegated
to the cohort implementation task's exclusive build target. No GPU, service,
installation, speed or release acceptance has occurred for this component.
