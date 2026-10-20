# Connected miss-only prompt ring

Kuno-only internal candidate, 2026-10-20. The selected installed worker remains
3fffc266 / SHAeac78ce0. No HTTP, KNT2/KNR2, configuration, model identity,
Vashti, release or installation change. Original DASLab IQ3_S/Q2 MTP stay.

## Reference and complete implementation

Pinned MIT Strata fb58e0 `sycl/src/prefill/prefill.cpp:3796-3850` advances raw
staging slots only for actual HOST experts, with an early return for resident
experts. The native branch3964-3983 records the slot-used event after both
dequantizations only when `slot >= 0`. Kuno's previous eight-group lookahead
instead advanced slots and split command segments for resident groups too.

The explicit Cargo feature `strata-prefill-miss-ring` / builder `--miss-ring`
requires the complete original native expert bulk path. It keeps every previous
connected dense/routed-wide, tail, attention, bulk/INFO, loader, lifetime and
transfer-dependency component. The three coupled ring changes are:

- Classify each routed expert from the CURRENT three effective Sources roles
  under the original immutable PromptLayer borrow. Original execution/graph
  checks run; loan shadows override base placement, and any HOST role means
  streamed. No stale resident mask substitutes for effective source ownership.
  Original per-group stage/source/registry/bounds guards still run, including
  resident groups. Prototypes and COMPLETE BDA read dependencies are unchanged.
- Assign eight original raw slots to actual misses only. Lookahead traverses
  resident groups without consuming slots. Every streamed slot is distinct
  among the eight staged, unconsumed groups. Logical consumption means original
  kernels were enqueued, NEVER GPU completion; actual original range/timeline
  dependencies still protect every physical overwrite.
- Bind resident groups to a genuine, checked, nonoverlapping same-size view
  in the existing32MiB raw-root padding. It preserves root allocation, stream
  and real descriptor bounds; all eight GU/DOWN format combinations fit nine
  such views. No GPU allocation or scheduler-only narrowed view is introduced.
  Separate real `ResidentGuImage`/`ResidentDownImage` operator identities have
  byte-identical original shader arithmetic and no streamed-slot release.
  Normal streamed DOWN still requires its original release marker. The closed
  grammar binds the new literal roles, requires matching resident/streamed
  pairs, and rejects reversed roles, mixed pairs and wrong release flags.

Original six logical phases, matrix pieces, full map/status/fault/INFO views,
physical source dependency inventory, model/loan/retirement checks, private
owner epochs/reuse rules, mutation guards, fences, erasure and unknown-flight
quarantine remain. KernelTask/Box release forwarding stays intact. Original
staging-copy validation still resolves/checks each role at its actual enqueue.

Ring/classification storage is bounded by512 groups and8KiB total temporary
HOST metadata, inside the already reserved24MiB whole-batch parent allowance.
The existing maximum route-conserving85-source budget fixture includes this
storage and still fits. Classification and slot vectors use full-capacity
Zeroizing owners; ring scalar state is cleared on normal/error/panic drop.
The original8MiB command tape still owns and wipes all collected arguments. The8MiB command-tape,4096 binding views,32 actual
pipelines,64MiB Graph cap and parent reservations are not enlarged.

Only the existing compiled-public profile publishes aggregate streamed/resident
group and peak-slot totals. There are no private prompt/route traces, persistent
request metrics, text/images, shader caches or additional network interfaces.

## CPU evidence and next gates

Actual combined App compilation and100 targeted CPU controls pass:
four ring/producer cases, two genuine serializer/budget cases (now both resident
and streamed producers),39 QSA,37 original queue/planning/submit/recovery policy,
three recording lifetimes, one actual scratch reuse, four transfer/peer-wait
controls, two profiler classes, four raw-slot controls, two vector-dequant
controls and two real KernelTask/Box metadata forwarding controls. A mistaken
filter initially selected zero tests; the actual `bulk_metadata_tests` filter
was run separately and both tests pass. Early test compilation caught two
fixture adapter mistakes (Handle getter and import path), fixed before these
final controls. No GPU or speed result is inferred from a CPU test.

The current-root actual runtime-source harness also passes all seven collector
and grammar controls, including original panic/escaped-client/changed inventory
guards and the new role/pair negatives. No GPU device, scope or completion
authority is constructed by that harness. Exact artifact-base logs:

- `native-miss-ring-app-cpu-20261020.log`
- `native-miss-ring-metadata-cpu-20261020.log`
- `native-miss-ring-runtime-cpu-20261020.log`

Release/frozen artifact, all ordinary/matrix offline layouts, original129 GPU
controls and complete real requests/counts/recovery remain next. Require a
meaningful clean matched gain before full staged thinking/tools/MAX8/semantic/
privacy, injected terminal/fresh recovery and official installation. Installed
matched prefix11.1474s and same-host Strata5.3515s remain the references; the
uninstalled preceding bundlec12a2418 repeat12.0792s is a diagnostic profile.

The first frozen builddf9fcf16 / SHA3ffea9a2 was NOT run on a GPU or installed:
source review identified the need to wipe the new routing metadata too. The
follow-up adds Zeroizing owners and a real still-owned-capacity erasure control
(no freed-memory inspection). Its five ring/producer cases pass, including the
new erasure case; together with previous cases this is101 App CPU controls.
Final frozen release and actual GPU/full-request evidence remain pending.

## Completed connected validation (d95a12da)

Frozen source `d95a12dab3ec137e90aa4e5e1099666415e9c71e`, worker SHA256
`07415e682c3fd8872a54546b93b365b5425c51274561b20adcce26fab450d41c`,
build `strata-parity-miss-ring-wipe-20261020`, passes release compilation,
408 ordinary/208 matrix offline layouts, all129 original GPU numerical controls
and three complete public requests. Both original runtime capacity/native-layer
proofs and the additional `miss-ring-validation.json` pass. No numerical gate,
real INFO/view/owner inventory, or parent reservation was relaxed.

Long repeated request:26,338 original groups =13,075 actual HOST misses +13,263
resident groups;39,225 role copies; peak-slot sum768 across96 phases. Actual
segments decrease48,851 ->36,513; rolling windows1,474 ->1,431. Profiled repeated
4245 prefix11.55808s versus complete predecessor12.07925s (4.3% shorter);
capture2.53357 ->1.95395s. Expert wall7.32303s, enqueue1.57346s, wait3.61396s,
QSA attention/merge1.46706s. These are diagnostic timings, not clean throughput.

Separate clean matched45/4245/4245 requests pass: load79.89410s, long first
13.99418s/repeat11.23059s. Installed first12.96228s/repeat11.14739s remains at
least as good, and same-host Strata repeat5.3515s is still faster. This complete
candidate stays **unselected and uninstalled**. No new full staged/terminal
acceptance or sustained decode gain is claimed.

The protected DSO-aware leaf CPU diagnostic also completes all requests and
physical recovery. Repeated4245 prefix11.68874s samples3.25126s user CPU:
memset0.54271s, Planner0.16080s, Wgpu storage lookup0.12060s, memmove0.10050s,
malloc0.09045s, interval split0.07538s, registry lookup0.07035s and snatch-lock
read0.05025s. Driver leaves are now distinguished from Rust leaves, but unresolved
addresses are not guessed into function names. Samples are leaf CPU estimates,
not inclusive wall or pure GPU costs.725,548B of raw sampling were deleted before
acceptance; only compiled-public aggregate metadata remains.

Exact receipts under the artifact base:

- `strata-parity-miss-ring-wipe-20261020-{operators,profile}/receipt.json`
- `strata-parity-miss-ring-wipe-20261020-profile/miss-ring-validation.json`
- `strata-parity-miss-ring-wipe-20261020-clean-long/receipt.json`
- `strata-native-miss-ring-dso-cpu-20261020/receipt.json`
- Owned summaries `strata-parity-miss-ring-wipe-public-summary-20261020.json`
  and `strata-native-miss-ring-dso-cpu-public-summary-20261020.json`.

All run cleanup snapshots show Intel50,688,000B, NVIDIA compute0, no new driver
faults, all three service endpoints200, admission inactive/unpaused/no recovery
and waiting0. The legacy clean helper's passed flag precedes cleanup, so its
actual host-after and final recovery marker were independently checked by
`verify-miss-ring-clean-recovery-20261020.py` (snapshot14:01:08UTC); DSO cleanup
snapshot14:10:17UTC passes too. Installed worker/gateway remain unchanged.
