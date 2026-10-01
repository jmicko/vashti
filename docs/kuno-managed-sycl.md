# Optional managed SYCL image worker

This is an internal Kuno runtime contract. The standard Kuno executable remains
independent of oneAPI, and its default image engine remains embedded CUDA/Vulkan.
The optional Intel worker uses the same private request pipes, preview framing,
cancellation, PCI selection and HTTP API. No Vashti consumer change is required.
GPU, performance and installed-service acceptance are separate from source tests.
This runtime family's ABI 1 worker was accepted locally for the original B70
generation and edit workflows after bounded IM2COL dispatch and wide copy
addressing fixes. The
large-decode installation block is superseded by the exact hardware receipts in
[kuno-image-performance.md](kuno-image-performance.md). It is an explicit optional
runtime, not a new default. The beta.2 distribution contains the historical ABI 1
worker; ABI 2 model retention requires a newly built and approved bundle.
Acceptance on other GPUs, drivers or Linux baselines is not established.

## Configuration

Existing image configuration needs no migration. Omitted `engine` means
`"native"`; omitted `runtime` means no optional runtime. These defaults remain
omitted when serializing old settings. To explicitly choose the optional worker:

```json
{
  "models": "/absolute/model/store",
  "pci_address": "0000:2d:00.0",
  "timeout_seconds": 3600,
  "collections": ["qwen-image-2512-fp8", "qwen-image-2.1-int8"],
  "engine": "managed_sycl",
  "runtime": "qwen21-sycl-2026.1-v1"
}
```

`managed_sycl` selects SYCL only for Qwen 2.1 text/edit requests. Qwen 2512 keeps
the embedded native route on the same configured physical PCI device. Collections
are never removed automatically. A gateway-only build may use the managed 2.1
worker, but cannot execute 2512 without the native image feature. SYCL advertises
only `standard` attention and rejects CUDA `fast`; its validated implementation
intentionally enables the backend's gated unmasked oneDNN attention route.

Configuration accepts only the fixed runtime ID above, never executable paths,
library paths, arbitrary IDs or request-provided runtime choices. Selecting
`native` with a non-null runtime is invalid. Config validation does not require
the runtime to be installed: a missing runtime degrades images without preventing
ordinary Ollama service startup. The privileged configure command verifies the
runtime before enabling it.

## Runtime manifest and trust

The fixed installed directory is
`/usr/local/lib/kuno/runtimes/qwen21-sycl-2026.1-v1`. Its flat manifest is:

```json
{
  "version": 1,
  "id": "qwen21-sycl-2026.1-v1",
  "worker_abi": 2,
  "ggml_max_name": 160,
  "engine_revision": "88411ef1e0688ff2df1010aeeb5d92b2d8cea2be",
  "ggml_revision": "4bf5f6000653b7881d00963cd6ddb665ccd62a8d",
  "engine_patch_sha256": "<64 lowercase hexadecimal characters>",
  "ggml_patch_sha256": "<64 lowercase hexadecimal characters>",
  "sycl_patch_sha256": "<64 lowercase hexadecimal characters>",
  "files": [
    {"name": "worker", "bytes": 123, "sha256": "<64 lowercase hexadecimal characters>"},
    {"name": "libsycl.so.9", "bytes": 456, "sha256": "<64 lowercase hexadecimal characters>"}
  ]
}
```

The numbers/hashes above are placeholders, not an installable bundle. The schema,
ID, 160-byte tensor-name size and revisions are exact. New model-retaining
bundles use worker ABI 2. The gateway also accepts already installed ABI 1
bundles, using their one-request-per-process path; it never sends them ABI 2
frames. Patch hashes are producer source attestations bound into the approved
manifest checksum; they do
not independently prove how the compiled worker was built. GPU/source acceptance
must record that separate evidence.

The resident engine's checked-in
`apps/kuno/patches/001-qwen-quantized-config.patch` source checksum is
`ac51db708fef026fe7abab850774309e4f1f439a05c56d06ff33e49fcc290584`.
This is a source-patch checksum, not an approved runtime manifest or worker
artifact checksum. Historical ABI 1 receipts below retain their original hashes.

Every member is a root-owned regular file, with no group/world write or set-ID
bits. The worker must be executable and have a 64-bit little-endian x86-64 ELF
header. All ancestors are root-owned directories without group/world write.
Symlinks, traversal, duplicates, unknown fields and unlisted entries are rejected.
Library names start with `lib` and contain `.so`; flat license/notice names use
`LICENSE-*.txt` or `NOTICE-*.txt`. Names have at most 128 ASCII alphanumeric,
dot, underscore or hyphen characters, with no `..`. SONAME/unversioned aliases
must be separately listed regular files. Installation materializes each alias
as an independent same-byte copy, so runtime acceptance must test that layout.

Limits: 64 KiB manifest, 2–96 members, 1 GiB per member, 3 GiB combined. Sizes and
SHA-256 values are checked using bounded reads. Nonregular members are rejected
without blocking on FIFOs. A gateway process caches a successful verification
only while protected inode/device, size, owner/mode and nanosecond modification/
change timestamps remain unchanged; directories are rechecked. Each fresh
launcher verifies the bundle independently before exec.

## Installation and rollback

Prepare a verified runtime-only bundle, including all observed lazy dependencies,
not the compiler/development SDK. Install it explicitly using an independently
approved checksum of the exact `manifest.json` bytes:

```sh
sudo ./kuno install-image-runtime \
  --directory /absolute/verified/bundle \
  --manifest-sha256 ACTUAL_MANIFEST_SHA256
sudo ./kuno configure-image --settings /absolute/image-settings.json
```

Each operation asks before interruption; `--yes` is explicit unattended consent.
The runtime installer requires root, verifies input before stopping services,
and uses the existing installation lock, ownership journal, atomic regular-file
staging and activation rollback. It does not enable SYCL or change image settings.
Later configuration/upgrades preserve the installed optional runtime, including
when the selected engine is native. Restore removes only journal-owned unchanged
files. Runtime artifacts are public code, with no prompts/images in the journal.
The installer currently reads replacement and rollback artifacts into RAM;
large bundles therefore need additional host memory during installation.
The transaction journal accepts at most 128 files, covering the gateway's seven
artifacts plus the runtime manifest and its 96-member bound. Oversized file sets
are rejected before creating a deployment journal or changing services; journal
writers/readers use the same bound so accepted installations remain recoverable.

The website installer still downloads/extracts only the standard Kuno binary.
Automatic optional-runtime download/publishing is not implemented here. Do not
publish the development SYCL-linked executable as the main Kuno binary: its
eager vendor dependencies would make ordinary gateway/upgrade commands require
oneAPI on every machine.

Release developers can produce a deterministic flat archive using
`apps/kuno/scripts/package-image-runtime.py --directory BUNDLE
--manifest-sha256 APPROVED_SHA256 --output ARCHIVE`. It verifies the exact pinned
schema, inventory, sizes and streamed hashes; rejects links/nonregular members;
and emits regular-file aliases, fixed permissions and zero archive timestamps.
It never overwrites an existing archive. The manifest checksum must already be
approved independently; packaging does not establish GPU correctness or source
provenance. No Python dependency is added to the installed service.
Native notice collection, prepared artifacts and release-owner steps are in
[kuno-release-handoff.md](kuno-release-handoff.md).

## Worker privacy and errors

The standard executable's hidden `image-worker-launch` entrypoint runs before
Tokio threads. It applies no-swap verification, zero core limit, nondumpability
and existing Landlock write denial, then execs the verified worker. Restrictions
therefore cover vendor constructors and their threads. The worker rechecks
privacy before reading any content. Exec can reset nondumpability; the zero core
limit and Landlock persist, and the worker restores nondumpability before stdin.
No additional writable filesystem exception is granted.

The worker receives a cleared environment with a fixed private library directory,
`HOME=/nonexistent`, existing CUDA/Mesa cache suppression,
`SYCL_CACHE_PERSISTENT=0`, `NEO_CACHE_PERSISTENT=0`, `SYCL_UR_TRACE=0`,
`GGML_SYCL_ENABLE_GRAPH=0`, `GGML_SYCL_USM_SYSTEM=0`,
`ONEDNN_VERBOSE=0`, `DNNL_VERBOSE=0`, `MKL_VERBOSE=0`, `XPTI_TRACE_ENABLE=0`,
`ONEAPI_DEVICE_SELECTOR=level_zero:gpu`,
`KUNO_EXPERIMENTAL_QWEN_SYCL=1` and `GGML_SYCL_FA_ONEDNN_NO_MASK=1`.
Preview framing is enabled only when the parent asks. `SYCL_DUMP_IMAGES`,
`SYCL_CACHE_TRACE`, preload/audit settings and inherited vendor debug settings
are absent. In particular, setting `SYCL_DUMP_IMAGES=0` is unsafe in the tested
runtime because its presence activates dumping.

Discovery marks affected Qwen 2.1 workflows unavailable using fixed reasons
`image_managed_runtime_unavailable`, `image_managed_runtime_invalid`,
`image_managed_runtime_incompatible`, `image_managed_runtime_untrusted` or
`image_managed_runtime_corrupt`. Ordinary Ollama routes and a ready 2512 fallback
remain usable. The same fixed reasons reach generation preflight as HTTP 503;
exec failure is classified `image_managed_runtime_exec_failed`. No private
paths, raw loader/backend exceptions, request content or credentials are logged.

## Acceptance boundaries

Source tests cover legacy defaults, runtime selection, 2512 preservation,
standard-only SYCL attention, ordinary Ollama proxy operation, manifest bounds,
checksums, ABI/header rejection, filesystem ownership/modes, symlinks/FIFOs,
cached identity invalidation, exact cleared environment, CLI confirmation/root
requirements and restoration of the full runtime file set on activation failure.
They do not establish a live library closure, GPU correctness, speed, privacy
under the vendor driver, cancellation/drain, device loss or installed activation.
Those require the root-owned hardware/service acceptance pass.

## Resident worker ABI 2

This is a private subprocess protocol, not a new HTTP API. The gateway embeds
the native worker and launches optional ABI 2 workers through the same protected
launcher. Requests use `KIR2`, one kind byte, a big-endian u32 payload length,
then payload. Kind 1 is a bounded JSON image recipe (at most 64 MiB); kind 2 is
an empty shutdown request. EOF is an orderly shutdown only between requests;
a partial header or payload is a terminal error.

Responses use `KIP1` with the same nine-byte header. Existing kind 1 previews
and kind 2 final PNGs keep their payload formats and limits. ABI 2 adds empty
kind 4 `READY`, empty kind 3 `CLEAN`, and kind 5 classified ASCII error codes
(at most 128 bytes). `READY` confirms the protected worker can read requests;
it does not mean model loading or sampling has begun. A final PNG is followed
by `CLEAN` only after native request cleanup and worker-owned Rust content
clearing. The supervisor validates the final PNG and waits for this
acknowledgement before returning success/releasing admission. An error ends
the worker; malformed, duplicate, oversized, or out-of-order terminal frames
cannot create reusable residency. Frames and raw errors are never logged.

One successful native model context stays loaded for compatible subsequent
requests. Qwen 2.1 generation/editing share its context; attention mode and
model family changes replace it. Prompts, negative prompts, references,
conditioning/prefix caches, latents, generated images and previews do not
survive as reusable worker state. Shader/vendor disk caches remain disabled,
and the existing no-swap, core-dump and filesystem protections apply throughout
the worker's longer lifetime. Idle shutdown uses the empty shutdown frame and
pipe closure so the native context destructor can release model resources.

The gateway retains at most one image worker and has no idle expiry. Managed
LLMs prefer other GPUs and share the image GPU only with an exact prior
in-process load profile plus sufficient live physical free memory. Unknown
shared workloads trigger orderly image eviction/reaping before ordinary load
preparation, and shared failures cannot classify retained image weights as an
unloaded baseline. Metadata leaves residency intact. Unverified-placement
fallback, model/runtime identity changes, cancellation, failure, recovery and
service shutdown release it. Global admission still serializes execution.

ABI 2 source implementation is separate from the historical acceptance below.
The existing ABI 1 timings and bundle checksums do not prove persistent model
reuse, cleanup acknowledgements, or speed improvements. New native and managed
ABI 2 hardware/service acceptance is recorded separately in the
[resident-image coordination receipt](kuno-vashti-coordination.md#resident-image-backend-receipt-2026-10-01)
and [exact build receipt](kuno-image-residency-build.json). Historical ABI 1
receipts below retain their original scope.

### Historical ABI 1 source receipts

Source receipt (2026-10-01 UTC): `cargo test --locked -p kuno -p kuno-protocol
-p kuno-image -- --test-threads=1` passed 182 tests with nine environment-dependent
checks ignored. Nonnative and ordinary native-image Clippy with all targets and
warnings denied passed.
The ordinary native-image all-target check passed against the performance
worktree's pinned `target/native-source` / `target/native-build`; no GPU was used.
Formatting and whitespace checks passed. That source-only subtask made no
service/runtime/GPU changes; later coordinating-task installation is below.

Review-fix receipt (2026-10-01 UTC): all seven `image_engine::tests` unit checks
and all 12 deployment integration checks passed serially, including a 43-file
runtime installation/rollback/restore and oversized initial/replacement sets
rejected without service or deployment-file changes. The exact worker
environment allowlist is asserted. Nonnative all-target Clippy with warnings
denied, formatting and whitespace checks passed. These are synthetic source
checks; later live acceptance is recorded separately below.


## Local acceptance and operational limits (2026-10-01 UTC)

The final verified bundle has 54 regular members, 531,417,369 bytes, manifest
SHA256 `94daf0ed8e7cd57ae861a0213c1ef5751a0261b7d2ee9d04bcd47d682941b59a`
and worker SHA256 `414b86a3c498c0f848321766e99784a8d5d7fd7f3984c1bf35796f6b5b14796b`.
Original full generation/edit, previews, two-reference editing and cooperative
cancellation pass with the complete SDK extraction inaccessible and only the
flat bundle libraries visible. See the performance receipt for exact geometry,
timings, memory peaks, copy/kernel proofs and remaining speed gap.

The ordinary CUDA/Vulkan gateway and this runtime were installed transactionally
with user authorization. The runtime installation preserved configuration byte
for byte. Explicit local configuration changed only image engine/runtime/PCI
selection to B70; collections, model store, listeners, Ollama settings and disabled
LLM response/idle deadlines were preserved. Global defaults stay native. The
first installed keyless 512x512/two-step stream passed two previews, final PNG,
IPv4/IPv6/LAN discovery/version and unauthenticated-control 401. Additional
installed service acceptance is recorded in the final performance receipt.

Changing image.pci_address also selects the native 2512 fallback's GPU; it is not
a per-workflow second-GPU policy. SYCL accepts standard attention only. Saved
recipes explicitly choosing CUDA fast must select standard; unsupported settings
are rejected rather than silently changing their requested computation. The
API permits up to ten references, but local final-worker acceptance covers two;
other counts/shapes must obey the existing bounds and are not proven to fit.

The current runtime is tested against this host's GLIBC 2.39 baseline only.
It is not a portability claim for older Linux versions. No driver/kernel was
changed for these fixes. New arbitrary GPU faults and indefinite stability are
not certified; ordinary device-loss/forced-exit handling retains existing
reservation/recovery semantics, and raw vendor exceptions remain filtered.

Transaction health checks establish gateway/admission service readiness, not
image execution, so installed image acceptance remains mandatory. Rollback
snapshots are in RAM during the transaction. Corrupted/missing journal-owned
runtime files cause configure/upgrade/recover ownership verification to refuse
before interruption; those artifacts require manual repair. Missing runtime at
gateway startup degrades affected images while preserving ordinary Ollama access.


Installed service acceptance also passed native Qwen 2512 on the B70, ordinary
Ollama generation, image-stream disconnect followed by immediate LLM inference,
and missing-runtime degradation/restoration without restart. Final admission is
resumed, service active/running, NRestarts 0, MemorySwapMax 0, no image worker or
recovery quarantine. No additional reset appeared during final acceptance.
Exact tests/timings and the bounded journal privacy check are in the installed
service section of [the performance receipt](kuno-image-performance.md).

## Prepared beta distribution receipt (2026-10-01 UTC)

The release archive is a notice-only superset of the hardware-tested installed
bundle: 64 members, 533,739,042 bytes, approved manifest SHA256
`bf31edfe7c9630b6b6b367c98831aa7d4e931a797fcf29d519b00d7843413a99`.
All previous 54 member bytes/hashes, including worker and vendor libraries,
remain unchanged. Ten additional native license/notice files complete the
identified embedded-source notices. The installed bundle and its historical
manifest checksum above were not changed. No release or runtime reinstall was
performed by this packaging work.

Prepared archive SHA256
`d242d7628ccb6c6048e7bdb7edca04d1c2e7453404e68667e829aff61b5ddd2f`,
156,166,814 bytes. Eight CPU packaging regressions and full archive read-back
passed. The release task must host the separate artifact and checksum before
advertising its download. Exact paths, source range, provenance, tested platform
limits and consumer/release steps are in [the beta handoff](kuno-release-handoff.md).
