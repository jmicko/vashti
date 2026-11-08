# Unselected exact persistent DASLab bank allocation

Status: source-only leaf, based on `42df907`. No controller, constructor,
admission ledger, source consumer or production selector uses this helper.
Five focused source tests, strict direct-device all-target Clippy and formatting
pass. Physical allocation, initialization-origin authentication and numerical
acceptance remain pending a separately integrated executing consumer.

## Three distinct authorities

`ExactBankQuote` is a metadata quote for one original DASLab layer, not an
allocation receipt or VRAM credit. Its six requests are the original gate,
up and DOWN `72+72` buffers. Original dimensions, raw types, row bytes and
complete tensor extents must match the pinned 48-layer inventory. The quote
uses the actual client's immutable memory alignment, rounds each root
separately and retains the 64 MiB resident-root bound. At alignment 256, the
complete 288-root quote is 14,144,716,800 bytes. Other alignments are computed,
never inferred from this observed or hypothetical value.

`ConstructionOrigin` is a narrow, private, unsafe construction-origin witness.
Its caller must create it at the fresh pinned Wgpu client initialization with
the same immutable `CubeClRuntimeConfig` Enabled snapshot. The actual client
must have no unsafe stream override, and no concurrent allocation, stream
mutation or allocator-mode change may occur on that client during the scope.
The global config and current stream alone cannot authenticate an arbitrary
already-existing client. The pinned API does not expose its private effective
stream or allocator initialization config; these remain explicit caller
obligations. The witness is bound to the client reference, config Arc, actual
alignment and current stream and is not Send, Clone or serializable.

`ExactAllocatedBank` is published only after actual allocation and validation.
It owns the real unchanged `ExpertBanks` and all six actual Wgpu buffer roots.
Every handle has the expected current stream, zero offsets and original full
logical size. Each resolved resource has offset zero, the original used span,
exact separately aligned backing and a distinct buffer identity. Those roots
remain retained with the bank. The six identities are distinct within this
one-layer publication; the future whole-model parent must still authenticate
all 288 distinct roots and their actual original source ownership. This
establishes declared buffer extents, not
opaque Vulkan heap requirements, loaded weight bytes, source attachment,
completed GPU computation, erasure, release or safe additional capacity.

## Allocation and accounting order

The unsafe origin requires the parent to hold its original exclusive completed
construction/admission boundary. Before allocating any bank root, the helper
passes the quote to the parent precharge callback, which must admit all six
exact requests while retaining every other original cap, debt and margin.
Its success is a caller-owned accounting obligation, not a fabricated receipt
from the leaf. The quote's checked `precharged_upper` helper only performs
arithmetic on a caller-provided original upper bound and limit; it does not
sample or replace `MainAdmission`.

Only MAIN is placed in the original CubeCL Persistent allocation mode. The
existing `PersistentPool` allocates each root at request rounded to the actual
device alignment; `WgpuStorage` creates a buffer of that declared size.
Staging and uniform pools retain their original mode and accounting. Ordinary
`PrivateArena`, `MainAdmission::page_bytes`, bank constructors, source layouts,
144-slot/512-expert maps, five GU/DOWN partitions, counters and defaults remain
unchanged. The bank's legacy `reserved()` remains its conservative bin bound;
only the separate owning publication can eventually inform a new admitted
consumer.

The persistent task is caught inside its scope. Auto restoration is attempted
on normal return, error and panic before returning. A panic, mode operation
failure, allocation error or returned-root mismatch retires the caller's
context. Validation after Auto restoration is also caught and retires on
panic. It grants no healthy reuse or release credit. The parent must perform
the existing full completed cleanup/close proof before any recovery credit;
dropping handles is insufficient. Restoring Auto does not stop the persistent
pool from reusing an exact matching size in later Auto mode. Other existing
allocation charges must remain conservative and the context must remain
serialized; no generic allocator-mode framework is introduced here.

The unchanged 512 MiB driver margin remains mandatory. Current v173 telemetry
observed 14,557,240,448 bytes of bank backing and 412,523,648 bytes of padding.
Its unused 4,166,132,608-byte conservative allowance is not free VRAM. A smaller
quote does not itself prove physical headroom. Root owns future protected
actual admission, peak observations, full original CPU13/repair, source/owner
closure and same-boot recovery before normal performance comparison.

## Source checks and remaining integration

Focused source tests cover all six pinned formats/all 48 layers and 288
requests, separate alignment/overflow/64 MiB bounds, checked precharge before
allocation, actual metadata handles with normalized owning offsets, wrong
full extents/stream,
config-policy rejection and Persistent-to-Auto ordering on success, error and
panic. A source check retains the resolved-buffer identity comparison;
distinct actual GPU roots remain a physical publication gate. Tests do not
construct a GPU or authenticate a real initialization.

Completed checks on 2026-10-11, with the locked native-text manifest and `-j2`:

- `cargo test --features direct-device --bin kuno-text-lab exact_persistent`:
  all five focused tests pass. They use the pinned public inventory fixture,
  metadata-only handles and simulated mode callbacks.
- `cargo clippy --features direct-device --all-targets -- -D warnings`: pass.
- Manifest-scoped `cargo fmt -- --check` and `git diff --check`: pass.

The exclusive build cache was the existing NVMe
`cpu-batch-native-target-20261011`; no GPU, service, device query, admission
activation, runtime reconfiguration or source attachment was performed.

The helper is intentionally unused. A future controller must join the actual
fresh initialization, snapshot, stream exclusivity, precharge callback and
owning result to the original admission/attachment lifecycle. No new slots,
six-partition geometry, user API, service installation or release is included.
