# Kuno beta release handoff

Recorded 2026-10-01 UTC (2026-09-30 local). **The tested backend is ready for beta
integration.** Overnight soak, ComfyUI parity and support on every Linux/GPU
combination are not requirements for this beta and are not claimed. Publication
and application acceptance belong to the Vashti release task.

**Integration update:** the main checkout is now on `codex/vashti-images` with
Vashti's earlier changes committed as `c6109d4` and the complete Kuno handoff
merged as `63ce405`. This handoff is available at
`/home/jmicko/dev/vashti/docs/kuno-release-handoff.md`; the historical checkout
state below describes the backend handoff before that integration. The verified
optional runtime archive has been copied to the main checkout's beta.2 release
directory. The source/build artifacts in the Kuno worktree are still needed and
have not been removed.

**Application release checkpoint:** flow/default fixes are committed as `20d2c0f`.
Vashti and Android `0.13.0-beta.2` (APK code
13002), standard Kuno, and the optional runtime have been built, checksum-checked,
and validated by the four-artifact signing dry run. Packaged Linux served the
expected version and final embedded frontend from a fresh data directory. Live
browser acceptance against installed Kuno is recorded in the coordination guide.
All four artifacts are published as `v0.13.0-beta.2` on the Hub and their full
public downloads, SHA-256 checksums, Ed25519 signatures, and prerelease update
manifests have been verified. Stable remains `v0.12.0`; no stable promotion or
live Vashti/Kuno binary deployment was performed. The approximately 644 MiB
upload used the authorized, verified 1 GiB Hub/nginx limits on 2026-10-01.
Physical Android testing remains outstanding; no device was attached.

## Resident image update

**Newer installed candidate, 2026-10-01:** `codex/kuno-image-residency`, based on
`4f9b465`, supersedes the one-shot image behavior and artifacts below for the
next beta. The published beta.2 receipts remain historical. Integrate the
complete resident branch; behavior ends at `4fb4e58`, build/source receipt is
`418569e`, and its newest receipt records installed hardware acceptance. The
main coordination guide points to the final commit. No Vashti API/UI changes
are required, but actual application/release validation still belongs to the
release task.

Artifacts live in `/home/jmicko/.codex/worktrees/kuno-image-residency/vashti`:

| Artifact | SHA-256 |
| --- | --- |
| Standard gateway `target/standard-cargo/release/kuno` (725,284,384 bytes) | `49115713860d966212ca09112f5166b126950c6b50e162e0fc4eb0a542fb9488` |
| Runtime `target/inference-tests/kuno-sycl-resident.tar.gz` (156,230,608 bytes) | `ee9c3e9caaf813a07d430e20598622f05b976e552390afe307bc5109a9a21ec1` |
| Manifest `target/inference-tests/runtime-flat-resident/manifest.json` | `12a901d727193a8419a318d17839396f50ada76195c5fda36f5cfd9e0ef76947` |
| Worker in that flat directory (135,499,920 bytes) | `5c1ea8f46669f5f1ea411134c6a30185fc0517b574d0996cea7eee14bad7a225` |

The exact [build and installed acceptance receipt](kuno-image-residency-build.json)
records pinned engine/patch/archive provenance and limitations. All 63 vendor
library/notice members remain byte-identical; both native engines and Rust
workers were rebuilt. Standard CUDA and installed private Intel ABI 2 retention
pass. New gateway accepts ABI 1 via one-shot fallback, but image retention on
Intel requires this ABI 2 bundle; do not reuse beta.2's runtime. Never ship the
private SYCL worker as the ordinary gateway.

Authorized local installation preserved configuration byte for byte. Installed
Intel generation/edit, previews, model handoff, cancellation/reload, idle service
teardown, legacy-family reuse and privacy/service checks pass. Service is
unpaused with Qwen 2.1 loaded. Small smoke timings do not establish ComfyUI
parity; Intel repeats have rounding-scale pixel drift, and no overnight soak or
new Android/Vashti acceptance is claimed. No next beta upload, merge or push was
performed. Keep this worktree/artifacts until the release task copies them and
runs its usual complete checks.

## Experimental performance candidates

The later graph-local RoPE and aligned contiguous-F16 K/V source candidates are
**experimental and excluded from release selection**. Their controlled resident
baseline, CPU checks, failed/pending GPU oracles and kernel-event boundary are in
[the performance appendix](kuno-image-performance.md#resident-full-workflow-baseline-and-experimental-speed-candidates).
The installed runtime/configuration and historical artifact receipts are unchanged.
Keep the packaging helper's current accepted pins until device recovery and
successful GPU numerical/full-workflow benchmarks, protected runtime/privacy,
and cancellation/reuse acceptance. A candidate build or passing source suite
does not authorize repackaging, installation or publication. No Vashti code/API
change or frontend integration is required for these internal candidates.

The sections below preserve the pre-beta.2 backend handoff. Their checkout,
artifact and release-action details are historical; the published beta.2
checkpoint and newer resident-image update above supersede them. Use the current
publisher and complete four-artifact checks for the next release.

## Source and ownership

Kuno worktree: `/home/jmicko/.codex/worktrees/kuno-image-performance/vashti`, branch
`codex/kuno-image-performance`, integration base `f6f6c22`. Integrate the **entire
linear range from that base through this branch's handoff commit**, not only the
recent SYCL commits. The exact final tip is also recorded in the shared
coordination guide. Behavior/hardware receipt commit:
`c5b45be7301209091cdd8387633fd1b837e290bf`; packaging/provenance/test commit:
`6ef9380f1787530913ac2b86970aeeaa018915bc`.

The main checkout `/home/jmicko/dev/vashti` remains on `codex/vashti-images` at
`f6f6c22` with uncommitted application/UI/migration/spec changes. Preserve and
commit that task's changes before integration. Preserve the untracked
`AGENTS.md` and `docs/kuno-vashti-coordination.md`; this task deliberately did not
stage them. Both coordination copies have a release receipt. Preserve the Kuno
worktree and ignored artifacts below until the release task copies them.

No Vashti/Hub code, version bump, merge, push, release upload or message to the
other Codex task was performed here. This handoff is not deployment authority.

## Contract and consumer impact

No inference API change: keyless image NDJSON previews/final results, discovery,
ordinary Ollama forwarding and privileged loopback controls remain as specified
in [kuno-images.md](kuno-images.md). Content remains transient in Kuno; Vashti
owns prompt/image storage. Long Ollama inference and cancelled-request drains
retain disabled default response/idle deadlines.

The default image engine remains `native`. Explicit configuration
`engine: "managed_sycl"`, `runtime: "qwen21-sycl-2026.1-v1"` selects the optional
worker for Qwen 2.1 generation/edit. Qwen 2512 uses embedded native on the same
configured PCI device. There is no per-workflow two-GPU policy in this change.
Local installed configuration selects B70 `0000:2d:00.0`.

SYCL discovery advertises `standard` attention only. Explicit `fast` returns
400 `image_attention_unsupported`; consumers must respect discovery and let
saved CUDA Fast recipes select Standard explicitly. Missing/invalid runtime
degrades affected images with fixed readiness reasons/503, preserving ordinary
Ollama and native 2512. No Vashti connector change is needed for this engine;
actual application release acceptance is still required.

## Prepared artifacts

All paths below are relative to the Kuno worktree, not the main checkout.

| Artifact | SHA-256 |
| --- | --- |
| Tested installed standard gateway, `target/release/kuno` (725,136,808 bytes) | `d2a7a293bb021daebb1dda2c8b21a67443881a10ce0787dbd4b7ae2c4946d8dc` |
| Optional runtime, `target/release-handoff/kuno-sycl-linux-x86_64.tar.gz` (156,166,814 bytes) | `d242d7628ccb6c6048e7bdb7edca04d1c2e7453404e68667e829aff61b5ddd2f` |
| Release runtime manifest, `target/release-handoff/manifest.json` | `bf31edfe7c9630b6b6b367c98831aa7d4e931a797fcf29d519b00d7843413a99` |
| Worker, `target/inference-tests/runtime-flat-v4/worker` (135,348,896 bytes) | `414b86a3c498c0f848321766e99784a8d5d7fd7f3984c1bf35796f6b5b14796b` |

The approved release runtime contains 64 regular members, 533,739,042 bytes,
plus `manifest.json`. It adds ten native license/notice files to the installed
54-member bundle whose manifest is
`94daf0ed8e7cd57ae861a0213c1ef5751a0261b7d2ee9d04bcd47d682941b59a`.
**Every previous member, including all executable/library bytes, is unchanged.**
Only the old bundle was installed/hardware-tested; the notice-only new bundle
was archive-tested. Do not confuse the two manifest checksums.

Native notice collection retains complete embedded miniz/STB/ZIP/httplib/JSON
copyright terms, JSON MIT permission text and standalone native licenses.
Existing vendor notices are retained. This is not a complete Rust dependency
SBOM/license audit. Standard packaging includes the same native notices and
CUDA notices when CUDA is enabled.

Companion `target/release-handoff/provenance.json` and `SHA256SUMS` are outside the
flat runtime. Do not put additional files inside the extracted runtime: the
installer rejects unlisted entries. Source-controlled producer evidence is
[kuno-sycl-runtime-provenance.json](kuno-sycl-runtime-provenance.json): source
pins, Cargo.lock hash, official SDK package URLs/hashes, compile/link flags,
reused object hashes, final archive hashes and complete runtime inventory.
Historical reused-object metadata is labelled; the final copy receipt/current
patch003 supersedes its older patch value. It is producer evidence, not a
reproducible-build guarantee. The developer link wrapper is
`apps/kuno/scripts/sycl-runtime-linker.sh`; exact source/CMake/ABI requirements
are in [the native build notes](../crates/kuno-image/native/sycl-dev-experiment.md).

To recreate the archive from the prepared public bundle without executing it:

```sh
python3 apps/kuno/scripts/package-image-runtime.py \
  --directory target/inference-tests/runtime-flat-v4 \
  --manifest-sha256 bf31edfe7c9630b6b6b367c98831aa7d4e931a797fcf29d519b00d7843413a99 \
  --output /absolute/new-output/kuno-sycl-linux-x86_64.tar.gz
```

The developer helper requires Python stdlib only; the installed runtime does
not require Python. The archive is flat, deterministic and contains regular
aliases rather than links. The helper refuses existing output files and rejects
changed content, invalid inventory/pins, symlinks and FIFOs.

After verifying the downloaded archive checksum, extract it into an otherwise
empty directory. From the ordinary standard gateway, installation is:

```sh
sudo ./kuno install-image-runtime \
  --directory /absolute/extracted-runtime \
  --manifest-sha256 bf31edfe7c9630b6b6b367c98831aa7d4e931a797fcf29d519b00d7843413a99
sudo ./kuno configure-image --settings /absolute/image-settings.json
```

The first operation installs but does not enable SYCL. Settings must contain
the existing model store/collections, chosen PCI and explicit engine/runtime;
see [the normative runtime specification](kuno-managed-sycl.md). Both operations
prompt before interruption; `--yes` is explicit unattended consent. Installation
needs root and additional RAM for replacement/rollback snapshots. Corrupted
journal-owned runtime files require repair before upgrade/configure/recover.

## Tested platform and beta limits

Tested Linux Mint host uses GLIBC 2.39, GLIBCXX 3.4.30 and working Xe/Level Zero
drivers for Arc Pro B70. No host driver/kernel was changed. Worker CPU objects
use AVX2/F16C/FMA/BMI2. Image privacy needs the existing Landlock/cgroup-v2
facilities and a verifiable zero-swap cgroup. Older systems/different GPUs are
unverified.

The current standard gateway has no oneAPI dependencies or RPATH, but its
CUDA-enabled ELF eagerly requires `libcuda.so.1`, plus Vulkan/GNU runtime
libraries. It was built with CUDA architecture 86, `GGML_NATIVE=ON`, VMM enabled.
It is a tested mixed NVIDIA/Intel-host artifact, not an Intel-only/general-CPU
portability claim. A Vulkan-only/no-VMM or older-Linux build would be a different
artifact requiring its own checks. Do not publish the SYCL-linked worker as the
standard gateway to work around this requirement.

Full Intel generation/edit pass but remain slower than ComfyUI. The API permits
up to ten edit references; live acceptance covers two and does not guarantee
every count/shape will fit. No indefinite stability claim or live induced
driver-fault test is made. These disclosed limits do not block this beta.

## Verification receipts

Detailed hardware evidence is in [kuno-image-performance.md](kuno-image-performance.md).

- Source: 182 tests passed, nine environment-dependent ignored; nonnative and
  ordinary native-image all-target Clippy with warnings denied; native release
  build, formatting and whitespace checks passed before this packaging change.
- Independent IM2COL and copy CPU/GPU numerical/address oracles passed, including
  actual VAE geometry and signed/unsigned 32-bit byte-stride boundaries.
- Flat worker, SDK inaccessible, fresh process/no swap/write denial: original
  1184x1776/50-step generation **153.950 s**, original 1248x832/25-step edit
  **115.513 s**, two-reference 512x512/four-step edit **61.282 s**, all previews
  and final PNG checks passed. Full edit cancellation cleaned up in 3.061 s.
- Installed HTTP: IPv4/IPv6/LAN discovery/version; keyless image previews/final;
  control 401; standard-only/Fast 400; native 2512; ordinary Ollama; disconnect
  cleanup followed immediately by LLM inference; missing-runtime 503 with
  ordinary Ollama 200, followed by recovery on restoration without restart.
- Final repeat after idle: 512x512/two-step stream, two previews and final,
  **40.870 s**, first preview 38.327 s. Client and worker used zero-swap cgroups;
  no generated images saved. Service active/running, NRestarts 0,
  MemorySwapMax 0; kernel Xe event count still seven with no new reset.
- Packaging: eight CPU tests passed, independently reviewed/rerun; real tar/ELF
  fixture proves inherited private Cargo targets cannot cause stale publication,
  build failures/private-worker gate reject; runtime deterministic read-back,
  equal-size digest corruption cleanup, symlink/FIFO rejection, no overwrite,
  approved manifest/ABI checks. Full 156 MB archive read-back verified all 65
  tar entries, exact inventory, regular types, sizes and hashes. Shell syntax and
  staged whitespace checks pass. No serving Rust code changed in packaging.

## Required steps in the Vashti release task

1. Preserve the application changes and both coordination documents; integrate
   the entire Kuno range from `f6f6c22`. Review spec conflicts with the application
   task's newer `docs/kuno-images.md` instead of overwriting either side.
2. Choose the next unused coordinated beta version and build the standard
   package with the normal compiler and reviewed native source/build. Clear
   private SYCL/linker variables and use `apps/kuno/scripts/package-release.sh`.
   It now fixes Cargo target/target-dir and reads that exact output, rejects Intel
   dependencies/RPATH, and collects notices. The old installed gateway checksum
   above is evidence, not the checksum of the future versioned release package.
3. To ship Intel support, add the separate runtime artifact to the publisher's
   artifact list and host its approved manifest checksum/install instructions.
   Before beta.2, the publisher assumed Vashti Linux/Android/Kuno standard only.
   The separate runtime publication was completed for beta.2 as recorded above.
   Existing generic Hub artifact storage alone is not proof the runtime is
   discoverable/installable.
   Keep inference API unchanged and make optional runtime selection explicit.
4. Test Vashti discovery, Generate/Edit, previews, Stop, multi-reference flow,
   saved Fast recipes on Standard-only discovery, and storage behavior against
   this backend. Existing browser receipts remain useful; physical Android and
   the final integrated application build are release-task checks.
5. Prepare every intended artifact and signed/checksummed release metadata
   before requesting the one-time upload key. Hub versions are immutable after
   publication; do not upload three files and try to append the runtime later.
   Follow the user's release authorization and normal publication workflow.

No additional GPU optimization or overnight soak is required before the beta.
If the standard-only release is chosen, describe optional Intel support as a
manual verified runtime install until its separate download is hosted.
