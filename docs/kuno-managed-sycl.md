# Optional managed SYCL image worker

This is an internal Kuno runtime contract. The standard Kuno executable remains
independent of oneAPI, and its default image engine remains embedded CUDA/Vulkan.
The optional Intel worker uses the same request pipes, preview framing,
cancellation, PCI selection and HTTP API. No Vashti consumer change is required.
GPU, performance and installed-service acceptance are separate from source tests.
This support is a draft: the private SYCL engine is not accepted for installation
or enablement. During root's 2026-10-01 UTC acceptance, the original edit recipe
failed before sampling and the kernel reported additional Xe reset events.
Do not enable this route until that failure is resolved and revalidated.

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
  "worker_abi": 1,
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
ID, ABI, 160-byte tensor-name size and revisions are exact. Patch hashes are
producer source attestations bound into the approved manifest checksum; they do
not independently prove how the compiled worker was built. GPU/source acceptance
must record that separate evidence.

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

Source receipt (2026-10-01 UTC): `cargo test --locked -p kuno -p kuno-protocol
-p kuno-image -- --test-threads=1` passed 182 tests with nine environment-dependent
checks ignored. Nonnative and ordinary native-image Clippy with all targets and
warnings denied passed.
The ordinary native-image all-target check passed against the performance
worktree's pinned `target/native-source` / `target/native-build`; no GPU was used.
Formatting and whitespace checks passed. No services, runtimes or GPU workloads
were installed, started or changed by this task.

Review-fix receipt (2026-10-01 UTC): all seven `image_engine::tests` unit checks
and all 12 deployment integration checks passed serially, including a 43-file
runtime installation/rollback/restore and oversized initial/replacement sets
rejected without service or deployment-file changes. The exact worker
environment allowlist is asserted. Nonnative all-target Clippy with warnings
denied, formatting and whitespace checks passed. These are synthetic source
checks; the live SYCL acceptance block above remains unresolved.
