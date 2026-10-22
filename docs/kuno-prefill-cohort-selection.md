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

Five focused CPU controls cover empty/incomplete grammar, exact actual scoped
population, foreign/stale owner and generations, changed Graph/node/range
counts, recording ID/family binding, forbidden queue/named/profile selection,
failure cleanup, bounded metadata and in-place erasure. Original threshold
controls remain unchanged. The positive selection case constructs all six
genuine grammar launches using actual Handle/Binding views and full physical
array metadata, calls the original completed grammar through `Book::begin`,
then closes and witnesses the actual appended span. Its metadata-only scope
fixture passes the original owner/generation/epoch/reuse state check and rejects
the stale generation before handing out the current scope. This does not replace
the actual device-backed `TransientIo` factory and owning-Graph hardware proof.
The original diagnostic profiling statement remains intact; authenticated
cohort profiling extends it in a separately feature-gated shadow binding.

For the existing authorized fixed-public layer profile, the feature exports
`NATIVE_COHORT_SELECTION_LABELS` and
`native_strata_cohort_selection_stats() -> [u64; 22]`. Collection is conditional
on the unchanged existing public GPU/host profile authorization; no private
request even contributes anonymous shape counts. The getter remains readable
after that window closes so the original collector can publish numeric deltas.
There is no new API, logging, persistence, route inventory or request identity.

The fixed counters distinguish genuine grammar admissions and logical cohort
groups/batches from their actual physical launches; complete Graph witnesses
and actual node/compute populations; successful rolling preparation from
successfully accepted rolling submissions; successful completed-pool retirement
and actual pool adoption; and successful recording re-encoding. Separate
below-1024 counters identify actual selection/submission/adoption under the
original size threshold. Existing generic rolling/pool counters remain intact.
Neither a witness nor rolling preparation claims completion. Retired-pool
credits occur only after the original joined-flight/reset/metadata checks and
successful pool ownership transfer; adoption credits occur only after the
original pool objects have successfully transferred into the new Recording.

Exact-source CPU execution now passes8 controls:4 selector controls (including
the genuine completed six-phase grammar positive),3 actual scope/state controls
and the unchanged original rolling-selector control. The direct-rustc harness
uses the real fully featured runtime grammar, verbatim original scope/state
and rolling code, and the full production selector module. Only original
public-profile/location observation wrappers are disabled. Source files match
frozen root `a75df1fb`; exact hashes, command, binary and limitations are in
`/media/jmicko/faststore/kuno-development/expert-cohort-selector-exact-cpu-20261022/`.
The harness creates no real TransientIo/device/Graph and grants no GPU acceptance.
Root full candidate passes196 App CPU/11 runtime and429 WGSL/280 offline layouts.
Actual device selection evidence remains required in both full profile chains;
zero below1024 observations may be legitimate and must not be fabricated.
No GPU, service, installation, speed or release acceptance has occurred for
this component at this checkpoint.
