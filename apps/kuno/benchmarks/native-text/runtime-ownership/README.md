# Kuno transient-I/O runtime patch (laboratory only)

This is an isolated, unpublished Rust experiment for Kuno's resident native
text graph. It patches four pinned CubeCL 0.10.0 crates. It does not change the
production Kuno/Vashti dependencies, public inference API, system services,
installed model files, the global Cargo registry, or any inference deadline.
Initial patch authoring was source-only. Protected device acceptance performed
later by the root task is recorded below; it remains a standalone laboratory.

The explicit laboratory `KUNO_LAB_INFO_BATCH_UPLOAD=1` requires compact INFO
slabs and uses one fully initialized mapped source page per uniform target page.
Records append in nonoverlapping aligned regions. A separate copy encoder
unmaps all pages and copies every full source before the task encoder is
submitted. Each submission closes its append pages and creates fresh owners;
in-flight parameters are never overwritten. The original hard runtime budget
precharges full source/target pages and every audit duplicate. All pages remain
tracked through normal erasure/inspection. Failed preparation keeps sources
tracked and unmaps them before the original eraser; it cannot publish a token
or a clean epoch. No implicit queue staging, pooling, public/private fallback,
extra main allocation or reduction in owning checks is introduced. The default
per-record path is unchanged. Complete fixed-public B70 acceptance now passes
for this mode, including all128 original native IDs and full normal owning
erasure. The shared/GDN threaded sample measures2.8263 decode tokens/s.
See [the complete-model receipt](../../text/receipts/next-shared-speed-20261008.json);
these timings do not establish production/private worker acceptance.

## Pin and licenses

`cubecl-common`, `cubecl-runtime`, `cubecl-core`, and `cubecl-wgpu` are copies of the published
0.10.0 crates. Their `.cargo_vcs_info.json` pins CubeCL commit
`7cf203735e095e640a2c03b2400d0faa03196bb4`, with each original `path_in_vcs`.
Original MIT/Apache-2.0 licenses and crate manifests are preserved. Wgpu 29.0.4
and all other dependencies are unchanged. `provenance.json` records the locally
cached published archive SHA-256, original/modified source hashes, and lock/patch
hashes. The patch is generated against those original archives, not mutable
registry source files. A cached archive's hash is a provenance pin, not a fresh
network authenticity check.

Use only these four `[patch.crates-io]` path overrides in the standalone lab.
Do not override production workspace dependencies or edit global registry files.
`transient-runtime.patch` applies to directories named `cubecl-common`, `cubecl-runtime`,
`cubecl-core`, and `cubecl-wgpu` containing the original pinned packages.

The optional lab Cargo feature `direct-device` enables the pinned common crate's
existing reentrant mutex device handle on native Linux. It executes server
enqueues on the caller instead of the dedicated channel thread; the original
server, stream, WGPU queue, resource leases and all erasure/error checks remain.
GPU command completion is still asynchronous until the existing fences; a
direct enqueue is not completion or acceptance. The feature is off by default,
requires explicit protected count1 `--device-dispatch inline`, and is checked
against the actual runtime handle's blocking property. The ordinary threaded
artifact and runner choice remain available. Collective/distributed device
operations are unsupported by this experiment. Protected B70 short and75/128
complete-model acceptance now passes with the actual blocking-handle marker,
original output IDs, fresh/cancel/recovery checks and full owning erasure.
Inline dispatch measures3.3473 tokens/s with standard banks and3.5006 with
intel96 in single ordered samples; source `1a80d5b`, normal frozen artifact
`fe490699…`. Production dependencies and default runtime dispatch are unchanged.

The later explicit cache-admission experiment permits independently pinned
canonical CPU caps0..10 in this same fixed-public count1 entry point. It does
not relax actual compiled-handle checks, private budgets, leases or erasure.
Intermediate-cap device acceptance is recorded separately in the owner
checkpoint; the original cap10 receipts remain valid historical measurements.

## Small public ownership API

The existing `ComputeClient` gains:

```rust
configure_transient_io(budget_bytes: u64) -> Result<(), ServerError>
create_private(data: Vec<u8>) -> Result<Handle, ServerError>
write_private(handle: Handle, data: Vec<u8>) -> Result<(), ServerError>
read_private_cleanup(handle: Handle) -> Result<Bytes, ServerError>
sanitize_transient_io() -> DynFut<Result<TransientIoReceipt, ServerError>>
sanitize_transient_io_audited() -> DynFut<Result<TransientIoReceipt, ServerError>>
```

Configure once on a serialized native stream while quiescent. Budget accounts
for FULL power-of-two spans, minimum 256 bytes, maximum 128 MiB per buffer. The 256-byte public minimum
satisfies map/copy/uniform alignment; these dedicated I/O spans do not inherit
the separate private main arena’s 32-KiB pool minimum. Driver allocation padding
is opaque regardless of that choice. No
private pooled uniform/readback pages or hidden retired pool pages are used.
Audited cleanup additionally charges one full MAP_READ stage for each upload
and uniform span; ordinary readback spans are inspected directly. Its required
budget is therefore original tracked spans plus upload/uniform spans. A
budget/map/verification failure fails closed and tries to erase any new stages.

`write_private` moves an owned Vec into a heap-only wiping controller,
without staging()/clone()/to_vec()/file controllers. Drop volatile-wipes FULL
capacity. The caller must first allocate and retain the full main handle, then
move a handle clone and its guarded padded upload. Initialization/reset/close
remain responsible for full main-span erasure. All private writes must use this
path, including selected embedding/PLE rows, statuses, routes and CPU-derived
vectors. Public weights retain ordinary create/create_from_slice APIs.

The convenience `create_private` allocates its destination internally and cannot
return that owner after a failed write. It is not used by the secure arena or
accepted as an integration substitute. Explicit preowned `write_private` is the
private worker contract. Misclassifying a private upload as public defeats the
boundary; there is no automatic content detector.

When configured, runtime scalar/metadata uniforms and readbacks use direct owned
buffers. Private writes/readbacks reject cross-stream bindings. Do not share
private handles across streams; root must serialize all graph enqueues and
lifecycle transitions. Unsupported backends reject these new APIs; no fallback
silently uses ordinary staging. Native Wgpu/Vulkan is the compiled target. Wasm
explicitly refuses the experiment; Wasm/no_std/other backend builds are untested.

`TransientIoReceipt` contains only `epoch`, full `upload_bytes`, `uniform_bytes`,
`readback_bytes`, and `inspected_bytes`/`inspected_buffers`. Ordinary sanitize
reports zero inspection counts. Audited sanitize returns acceptance only after
checking all original spans plus its own audit staging. Its inspected bytes
should equal the sum of the three per-kind totals. Contents/bytes/patterns are
never returned or logged by the audit.

## Ownership and cleanup order

Private upload staging uses MAP_WRITE|COPY_SRC mapped-at-creation buffers. Full
API-visible spans are initialized before a logical copy. It bypasses
Queue::write_buffer[_with]; Wgpu's native mapped-at-creation MAP_WRITE branch
maps the original buffer directly, whereas its non-MAP_WRITE branch constructs
an implicit StagingBuffer. Uniforms use directly owned UNIFORM|STORAGE|COPY_DST|
COPY_SRC buffers, with full clear then owned upload. Readbacks use directly owned
MAP_READ|COPY_DST buffers, with full clear before source copy. Bounds include
checked alignment, offset overflow, and actual backing-buffer size.

Returned readback futures hold cancellation/unmap leases. Successful readbacks
copy directly into guarded heap controllers, whose FULL heap capacities wipe
on Drop and whose leases survive until returned Bytes drop. Pending future Drop
aborts mapping via unmap; a sanitizer refuses live leases before flushing or
touching buffers. Resolve/drop every future and Bytes before cleanup. Copying
Bytes into ordinary host Vecs creates additional caller-owned private copies;
those need their own guards. No raw mapped or request bytes escape the audit.

For explicit cleanup, stop new enqueues; resolve/drop readbacks; checkpoint
transient I/O to free its bounded budget before eraser uniforms; enqueue full
main private-arena erasure; then use the checked sanitizer to drain those erasers
and erase their transient I/O. Accept/release main spans only after both stages. The
sanitizer flushes the unsubmitted encoder, performs a checked native queue drain,
remaps and volatile-wipes FULL upload spans, unmaps to flush, clears FULL uniform
and readback spans, submits/waits, and accepts scoped OOM/Internal/Validation
errors. An audited sanitizer then inspects every erased full span before
release; upload/uniform spans are copied to tracked direct MAP_READ stages,
while original readback spans are mapped directly. It then erases ALL original
and audit spans again and accepts completion/errors before advancing its epoch
or releasing buffers. Failing audits still attempt that final erasure.

Native drain/map/erase polls have a 30-second limit PER POLL, solely at cleanup
and configure/audit boundaries. This is not an inference timeout. Ordinary
stream sync/readback inference behavior is unchanged. Root must not call an
unbounded ordinary `client.sync()` first when closing the private arena;
integrate its final erase/drain transaction with this checked boundary.
The cleanup guard starts BEFORE scheduler execution, skips ordinary submission
regulation only at this boundary, and uses bounded public-write drains there.
Live leases reject before scheduled tasks/encoder mutations. Failed or panicking
cleanup/private operations permanently retire their stream even if the channel
catches a panic and keeps the server object alive. Many
spans mean multiple bounded polls, so total cleanup time is not a single
30-second deadline. Native Wgpu pop_error_scope returns an already-ready future.

### Bounded audit copy/map batching candidate (2026-10-06)

The source candidate keeps the same initial full-span erasure, full zero
inspection, and final full-span erasure before ledger advancement or release.
It charges and immediately retains the same one complete MAP_READ verification
stage for every original Upload/Uniform owner. Original Readback owners are
inspected directly. All verification clear/copy commands now share one encoder
and one submission; the subsequent checked map poll drains those copies too.
No separate per-copy submission poll remains.

Readable spans are inspected in fixed groups of at most 64. Fixed guard slots,
buffer references and typed callback channels are prepared for the entire group
before any map request. Every requested map owns an unmap guard before
`map_async`. One unchanged 30-second deadline per group covers preparation,
requests, the checked native poll and every callback delivery. Every callback
must succeed before any full-span scan in that group. A queued success delivered
after the deadline still rejects. Rejection, disconnection, timeout, nonzero
padding, allocation/copy/submission error, and panic release no accepted epoch;
pending/successful maps unmap before the existing final erasure attempts every
retained original and new stage. Typed first-observed failure and scoped GPU
errors remain enforced. A synchronous driver call or descheduling can still
overrun the requested wait bound.

Inspection still requires each mapped view to equal its complete buffer size
and contain only zero bytes. Source/stage counts, the separate runtime ledger,
main admission, leases, public weights, ordinary inference waits and external
APIs are unchanged. This first checkpoint batched only audit copies/read maps;
the subsequent source candidate below batches eraser WRITE maps while retaining
the same full-span volatile writes. No privacy setting, raw-data export or new
metrics are introduced.

Added source fixtures use the actual shared preparation and mapping policies
for 63/64/65 boundaries, multiple/partial groups, all-callback-before-scan order,
full original/duplicate counts, retained charges, partial allocation/copy,
submission, callback rejection/disconnection/late delivery, map/poll failure,
panic, nonzero tails and unmap retirement. Formatting and provenance generation
are source checks only. Root subsequently compiles and executes45 runtime host
fixtures and436 native host/offline tests with strict Clippy. Both cards pass10
GDN,52 routed/14 packed-format/12 expected failures, eight staged CPU-miss cases
and the complete eight-token native model/cancellation/recovery/full owning
erasure gates with the normal `91d4dd9` binary. See the
[owner checkpoint](../../../../../docs/kuno-native-text-checkpoint.md#batched-audit-and-staged-cpu-miss-acceptance)
for exact artifact identities and short timing observations. Matched long-run
performance and production integration remain unaccepted.

### Bounded upload eraser map batching candidate (2026-10-06)

The next source candidate preserves both full erase passes and the accepted
audit policy above. Upload owners now use fixed groups of at most 64 WRITE
maps. Every available full owner/channel/unmap guard in a group is prepared
before its first map request. Each group shares one unchanged 30-second deadline
across setup, requests, one checked native poll and every callback delivery;
there is no per-owner map retry or renewed deadline. No uploads means no group
poll. Each successfully mapped view must equal its complete API-visible buffer
size before every byte is written through the existing volatile wipe and
compiler fence. Full mapped views drop before unmap.

Erasure remains best effort across every owner. Individual preparation,
request, poll, callback, view/wipe or unmap error/panic is recorded without
skipping later available owners, groups, destination clears or final submission
and scoped-error acceptance. Successfully delivered callbacks can still grant a
best-effort wipe after a different owner's failure; they cannot grant a clean
transaction. Every requested/pending owner gets an explicitly checked unmap
before later groups or destination commands. Guard Drop retains the fallback
unmap obligation during failures/unwind; unmap failure permanently retires the
stream. The first typed callback/poll/failure category remains authoritative.

One failure prevents epoch advancement, receipt publication and buffer release,
even if later operations clear all bytes. Owners, full spans, charges, scope
futures, initial drain and final checked submission remain unchanged. Audit
verification copies/read maps are untouched. Ordinary inference/readback waits,
external APIs, configuration and public weights are unchanged; this affects
only explicit cleanup and the existing best-effort owner Drop cleanup.

Four source fixtures exercise the actual shared upload group policy at
0/1/63/64/65/129 boundaries, full sentinel tails, all-guards-before-requests,
callbacks-before-wipes and all unmaps before later groups/clears. They inject
nth-owner setup/request/callback/wipe/unmap errors and panics, poll error/panic,
rejected/disconnected/missing/late callbacks, wrong view extent, invalid/missing
group owners and partial final groups. Assertions require later-owner wipes,
every requested unmap, both destination clears/final submission, typed first
failure, retirement and unchanged failed epoch/charges. Existing audit fixtures
remain intact. Root subsequently accepts49 runtime fixtures,436 native host
tests, strict Clippy and provenance, plus the same both-card operator and
complete eight-token model/cancellation/recovery/erasure inventories using the
normal `200b38a` binary. Exact artifacts and short observations are in the owner
checkpoint linked above. No matched long-run timing or production authority
follows.

Any cleanup error or panic retires the private stream, retains tracked buffers
WHILE the owner remains live, returns no success receipt, and requires retiring
the whole shared context/poisoning its sequences. A future cannot cancel cleanup
midway: cleanup is synchronous at the server submit boundary. Explicit cleanup
is authoritative. Drop attempts best-effort cleanup; if it fails, eventual
object/context Drop may still release dirty resources. Do not claim erasure
because objects were dropped or because a device was retired.

## Host scalar/INFO patch

Runtime ScalarArg values remain runtime inputs; private values must NEVER be
#[comptime] or enter persistent shader/cache keys. ScalarBuilder, both U32/U64
MetadataBuilder states, and packed MetadataBindingInfo wipe all retained Vec
capacities on finish, abandonment, and unwind. Guarded POD growth wipes old
allocations before release, rather than ordinary Vec reallocation leaving old
private bytes behind. A guarded packed Vec covers partial pack panic and the
Wgpu conversion temporary. Packed Debug reports a word count, never values.

The retained std INFO TLS now has a same-thread launcher lease; Drop wipes and
releases it on cancellation/abandonment/panic. Nested same-thread launchers
reject rather than mix INFO. KernelLauncher becomes !Send/!Sync. Invalid TLS
borrowing fails closed without double-panicking; it retains the busy marker
until worker-thread teardown, whose member destructors wipe. Ordinary owning
launcher unwind releases its RefCell borrow before lease cleanup. SCOPE TLS
keeps types/IR only; request-dependent comptime values remain forbidden.

## Host verification

```sh
cargo check --offline --release -j 2 --target-dir /tmp/kuno-transient-runtime-patch/target
cargo test --offline --release -j 2 --target-dir /tmp/kuno-transient-runtime-patch/target --lib
cargo clippy --offline --release -j 2 --target-dir /tmp/kuno-transient-runtime-patch/target --lib --tests -- -D warnings
```

The repository host harness passes 28 tests and strict Clippy, importing the
actual modified source modules. Its fixtures cover full-capacity heap/scalar/metadata wiping, retained-builder partial
pack panic, abandoned/panicking TLS launchers, fail-closed invalid TLS borrowing,
owned upload pointer movement, full-span bounds/checked budgets/epoch overflow,
lease counts across cancellation/returned ownership, shared native/fake erase
ordering, failures at map/copy/drain/completion, audited full padding/counts,
extra-stage budget charge, failed audit budget/map/nonzero padding, and
LIFO nested-error-scope cleanup on early return/panic. The fake
backend verifies release acceptance only after the shared policy and leaves a
separate public cache untouched. No adapter/device is initialized. This is not
the complete upstream CubeCL test suite or live GPU privacy acceptance.

## Required protected device proof / remaining limits

1. On BOTH physical GPUs, configure the experiment with a bounded ledger; upload
   synthetic nonzero/private patterns including stale logical tails; launch with
   runtime scalar values; read/return/drop guarded Bytes; audited cleanup must
   inspect every tracked FULL span and return zero acceptance with exact counts.
2. Verify public packed weights/expert-cache remain usable and model parity
   survives a successful audited checkpoint. Run token/cancel/reset/retry paths
   through whole-sequence main-span + transient cleanup and fresh-sequence parity.
3. Exercise pending/dropped/retained futures and Bytes: live leases must reject
   sanitizer without mutations; after leases drop it must clean. Inject map,
   copy, drain, completion, audit-nonzero, and audit-budget errors; no epoch reuse
   or accepted receipt, complete owner/context retirement, and best-effort
   cleanup of newly allocated audit stages. Do not dump private bytes on failure.
4. Audit ALL root host request allocations, preceding casts/copies, packed data,
   CPU PLE state, final logits/sampling, and uploads. Disable swap/core dumps and
   profiling/tracing for protected fixtures. Source/compiler cache input must
   remain public/static. This runtime patch cannot scrub earlier caller copies,
   stack/register spills, allocator historical data outside its guards, or
   unrelated processes.
5. Even live success proves full owned API spans under checked Wgpu operations;
   it does NOT prove scrubbing of opaque physical-allocation padding, Vulkan
   driver caches, firmware memory, CPU swap/core files, or abandoned context
   allocations. No blanket production privacy guarantee follows from Drop,
   receipts, fake tests, or compilation. Document these boundaries precisely.

The earlier standalone I/O checkpoint did not connect the complete graph.
The protected public resident graph now uses owned staging and passes both-card
chat/cancellation/fresh-owner acceptance. General host request ownership,
persistent private worker/lifecycle and production privacy remain unaccepted.
No release or production API is part of this laboratory.

`read_private_cleanup` is a narrow cleanup-boundary API. It retains a full,
untrimmed, same-stream owner, checks that its actual API backing has no hidden
tail, enters the cleanup guard before scheduler mutation and uses checked native
map polls. Ordinary read/read_async polling is unchanged. Its guarded returned
Bytes must drop before I/O sanitization. The optional arena zero audit inspects
one full main page at a time, drops its host lease, audits/erases readback staging,
and releases no main page until the whole owner succeeds. Failed audit Drop
quarantines bounded retained handles until process exit; it emits no clean receipt.

Both cards verify 1,286 main pages/785,186,816 bytes for each of four fixed public
owners, including queued GDN/QSA cancellation and fresh recovery. The separate
64-MiB runtime ledger stays in force. Optional token-boundary cleanup retains
intermediate tracked spans under that cap, while completed reads credit only the
main pool; final fault/logit acceptance sanitizes before any state commit.
Exact receipts are in `rust-secure-resident-owned-io-20261005.json` under the text
receipts directory. Root checks pass 28 imported runtime host tests and 169 native
host/offline tests; independent exact-reconstruction runtime/arena checks pass
43. Current composed patch SHA256 is
`a0d985c9b32653e95289cd64af651208f7b4b6142eb91e55e2af08e29e9ec5eb`.
Owned API pages are the proof boundary; opaque driver/firmware memory, old caller
copies/stack spills, historical swap and abandoned contexts remain excluded.


## Repository verification entry points

### Current mapping/slab follow-up (2026-10-05)

The current composed patch adds static, content-free first-observed failure
categories and one shared mapping deadline at eraser upload, audit and cleanup
read seams. Waiting for callback delivery repairs a source-confirmed race with
the background poller; it does not retry mapping, relax erasure or introduce an
ordinary inference deadline. Native polling and callback delivery consume the
same30-second requested-wait window. A synchronous driver call or descheduling
may overrun that requested bound. There are37 whitelisted category codes; no
native error descriptions, request bytes or scalar values are formatted.

The default-off target-only INFO slab is append-only within one epoch and
reserves complete source/target spans plus duplicated future audit headroom
before mutation. It neither pools mapped sources nor reuses a target across
epochs. Valid sizes are0/4096/8192 bytes; absent means0. Oversized records retain
the checked individual path. Full spans including unused tails stay owned until
accepted erasure. The independent runtime64-MiB cap and main ownership budgets
are unchanged.39 runtime host fixtures pass. All sizes pass fixed public
operator tests on both GPUs;0/4096 also pass full chat, explicit cancellation,
poisoned retry rejection, fresh recovery and four complete owner audits/card.
The long96-slot public generation also accepts cleanup after the callback repair.
These are laboratory receipts, without production private-worker or timing
selection. See [the measured limits](../../../../../docs/kuno-text-performance.md#mapping-callback-completion-repair-2026-10-05).
The current18-file composed patch reconstructs exactly against pinned original
archives; its SHA256 is
`64e243409cde4685bc0207876156b595e0081b8d830c1faf66117bb1deb8278f`.

From the repository root, use the standalone host manifest at
`apps/kuno/benchmarks/native-text/runtime-ownership/host/Cargo.toml` with
`cargo test --locked --offline --lib` and Clippy `--lib --tests -- -D warnings`.
Private-arena ownership/cleanup-policy tests also run in the normal standalone
`native-text/Cargo.toml` laboratory suite. No device is initialized by host tests.
`regenerate-provenance.py` regenerates the composed patch and all modified hashes
against the three exact cached published archives. Whole patch application to
fresh original archive files passed. A cached pin is not network authenticity.

The opt-in `transient-io-probe` uses only fixed synthetic patterns and runtime
scalar uniforms. It checks pending/retained lease rejection, dropped future
cleanup, full-span audit counts, public cache preservation, repeat use and main
span zeroing. It accepts no arbitrary prompt or model input and writes only
explicit public count receipts when an operator requests them. Device acceptance
and remaining failed attempts are recorded in the residency/performance guides.

## Both-card owned-I/O acceptance (2026-10-05)

Frozen Rust binary
`9136b527e6634fe4394076b8bce297bb2a5a4aee6775385461fc418d5ee057e7`
passes the fixed `transient-io-probe` on B70/Vulkan and RTX3060/Vulkan. The
operator units have MemorySwapMax=0, LimitCORE=0, MemoryMax=4G and a
180-second test bound. The main and I/O budgets are 1 MiB and 8 MiB. The
8193-word pattern occupies a full 65536-byte main span. Counts-only receipt:
`../../text/receipts/rust-owned-io-both-gpus-20261005.json`.

Pending futures and retained returned Bytes reject sanitizer entry. Dropping
them permits audited cleanup; each epoch's inspected bytes equal the sum of
its complete upload/uniform/readback spans, including audit stages. Runtime
scalar changes preserve kernel parity. An unrelated immutable public allocation
survives repeated cleanup. Closing the secure arena erases its full span,
rejects stale IDs, and leaves the context usable. No private bytes are returned
by the audit or recorded by the harness.

An earlier constructor rejected explicit zero offsets emitted by the actual
contiguous allocator. Validation now accepts absent/zero offsets equivalently,
rejects every nonzero trim before size arithmetic, and is covered by actual
allocator fixtures. This was an upload-validation failure, not a GPU fault.
The two later host-fixture ownership assertions were corrected to retain an
independent memory reference; they did not change the accepted runtime logic.

This proves erasure of tracked API-visible spans under checked operations. It
does not prove opaque physical driver padding, firmware memory, historical
swap, complete resident graph privacy or production worker acceptance. The
composed patch applies to fresh pinned archive sources and all 18 changed-file
hashes match `provenance.json`. No global registry, installed service, public
API or ordinary inference deadline changed.


## Fixed diagnostic categories (2026-10-05)

Owned I/O records the first observed infrastructure failure as a fixed enum code,
shared across future/controller/cleanup lifetimes. GPU error variants are matched
without formatting their description or source. The executor forwards only the
28 exact whitelisted codes; unknown descriptions become the existing generic
cleanup error. Held-readback rejection remains recoverable and does not latch a
failure. No cleanup attempt, poll bound, allocator or retirement policy changed.
31 host fixtures and204 native tests pass. The protected default-off96-slot long
run exposed `upload_map`; further callback/error-scope diagnosis remains open.
This diagnostic is not evidence of successful erasure after failure.

## Owned public weight upload candidate (2026-10-07)

`ComputeClient::create_public_owned(Vec<u8>)` is a narrowly scoped laboratory
API for immutable original public model bytes. It moves the caller's existing
Vec into `Bytes::from_elems` and the ordinary server upload queue, avoiding
`create_from_slice`'s slice copy, the internal vector copy and possible aligned
allocation copy. The original contiguous layout policy, one-byte geometry,
stream selection, memory initialization and ordinary `server.write` are
preserved. GPU staging/upload still copies; this is not zero-copy device I/O.
The queue owns the payload until the server processes it.

Only the ordinary original GGUF static-weight path and individual public expert
source upload call it. Both callers already supply fully initialized padded
public payloads. Explicit ordered-public source mode, reused bulk-fill buffers,
private activations/uniforms/router packets and every erasure/admission path are
unchanged. It must never receive request-derived content or routing metadata.
The type system does not classify public versus private bytes; this boundary is
enforced by the narrow reviewed call sites, not a general secure Vec facility.

Source checks pass459 native tests,49 runtime tests, all-target strict Clippy and
formatting. Both-card original-blob routed operators and complete model, fresh
replay, cancellation, poisoned retry/recovery and owning zero audits are required
before physical acceptance. A CPU leaf sample led to this investigation but
does not isolate these copies or establish a performance gain.

Both-card physical acceptance is now recorded for normal source `79ba326` /
`9d4f4277…`:52 routed cases,14 formats,12 failures and the original complete
short model/fresh replay/cancellation/retry rejection/recovery/full owning audits
pass without retirement. See
`../../text/receipts/rust-native-owned-public-upload-device-20261007.json` and
[the latest checkpoint](../../../../../docs/kuno-native-text-checkpoint.md#latest-owned-public-upload-and-long-comparison-2026-10-07).
The accepted long Intel Q4 comparison uses this upload path in both profiles;
it is not a copy-before/after benchmark or a production routing acceptance.
