# Whole-RoPE release-owner checklist

The early source-only checklist below is historical and superseded by the
authorized source selection and installed owner receipt in
`docs/kuno-release-handoff.md`. It records how this closure was reviewed.

The original source-only handoff compared accepted residency `fb2d1d0e367d4d53054bf8de63f49b316046d704`
with candidate `0c646feb94c471ce579f235bec88a6867a491bc1` (production checkpoint
`230d15b`). It does not select source, build defaults, package pins, an installed
runtime or a release. The parent is running the counter-free full comparison;
that outcome is separate from this source review.

## Exact production delta

Only three tracked files affect production compilation relative to accepted residency:

| File | Resulting behavior |
| --- | --- |
| `apps/kuno/patches/001-qwen-quantized-config.patch` | Qwen21 prepares one contiguous positional matrix per compute graph and shares it across Q/K layers; adds common Rope helpers with the legacy operation path retained. |
| `apps/kuno/patches/003-sycl-convrot.patch` | Adds aligned dense-F16 K/V aliases, guarded residual-first F32 MUL→ADD and default-off whole-RoPE matching/kernel. Existing ConvRot and other accepted sections retain their bytes. |
| `crates/kuno-image/build.rs` | Adds source/header rerun paths; complete source-pin/diff, cache and ABI verification remains unchanged. |

The remaining sixteen changed files are tests and documentation. Patch002,
Rust worker/gateway/application implementations, protocol, frontend, shim,
Cargo.lock and package scripts are unchanged. Exact baseline/candidate hashes,
Git blob IDs and native patch sections are recorded in
`target/whole-rope-residual-v1/release-handoff-source-review-v1.json`, SHA256
`2f426f4141ac32eaef18ee845f17e9738fa1a4039ec2a54f361eb6af19e6542c`.

Production history is attention `c04ab15` + alignment guard `f6aa880`, shared
engine PE `0cec056`, residual introduction/guard/arithmetic changes
`4d603ab`, `ecd532f`, `4296d4a`, `7579f98`, and whole composition `230d15b`.
These overlapping patch commits are provenance, not an instruction to cherry-pick
a subset blindly. Integrate the reviewed final file tree with its relevant
validation coverage. `12d3687` and `0c646fe` are checkpoint documentation only.

## Source and build choices still requiring explicit selection

| Pin | Accepted helper today | Experimental candidate |
| --- | --- | --- |
| Engine revision | `88411ef1e0688ff2df1010aeeb5d92b2d8cea2be` | unchanged |
| GGML revision | `4bf5f6000653b7881d00963cd6ddb665ccd62a8d` | unchanged |
| Patch001 | `ac51db708fef026fe7abab850774309e4f1f439a05c56d06ff33e49fcc290584` | `ed75739ccb193099f23d4600cc4a8846f9d17fe6c2cec7591a7c9cb30c2d0134` |
| Patch002 | `7298590faed44cad4733c78f6f6c2bf261b647712b5ec5f75840da450f0067dc` | unchanged |
| Patch003 | `7f6a75c75544c0e3776d780b0710cab2fabb94ddfe57eeb436dff02891bd4771` | `3d1191da4eb004a25c52116fa30421c3a7b617fc4e854d416248b2d7b0070794` |

Whole-RoPE's CMake condition is false when its variable is unset. The private
build explicitly passes `-DKUNO_SYCL_WHOLE_ROPE_BYPASS=ON`, corresponding to the
production compile definition `KUNO_SYCL_WHOLE_ROPE_BYPASS=1`. No published or
selected build default has been changed. A release owner must explicitly choose
ON in the normal native build recipe to reproduce this worker's behavior. Keep
whole's precise/no-fast-math/no-associative-math/contract-off source flags, graph
capture OFF, and omit WHOLE_TEST, repeat bypass, profiler and link wrappers.
No guard or precision change is part of this handoff.

Patch001 is global. Qwen21 uses the shared prepared matrix on CPU, CUDA, Vulkan
and SYCL; the whole kernel itself is SYCL-only. Selecting this patch therefore
requires rebuilding and validating the standard embedded CUDA/Vulkan native
engine and standard gateway, even though their GGML patch002 is unchanged.
The source verifier alone does not prove an old archive was rebuilt. The private
SYCL closure's engine56fc reuse is justified by the independent 57 C++/211 source
and full-public-header proof; it is not new CUDA/Vulkan hardware acceptance.

`package-image-runtime.py` PINS are still the accepted values above. The ignored
private `target/.../package-runtime-v1.py` substitutes the reviewed candidate
pins in memory only; it is not a selected release packager. After acceptance,
the release owner must deliberately reconcile official packaging PINS and final
manifest/build-source receipts. Runtime validation checks fixed revisions/ABI,
root ownership and member integrity, but only requires patch hashes to be valid
hex; it does not independently select candidate patch hashes.

## Required release receipts

- Choose the exact final source/pins and whole activation; retain meaningful CPU,
  failure, topology, finite/special-value and native exact oracle coverage. Current
  combined source passed CPU whole79/1372, nine failures and actual-engine cleanup;
  the parent reports merged GPU whole86/1400 and residual48 exact passes.
- Produce normal-verifier native source/archive/flag receipts for every distributed
  backend. Rebuild the standard embedded engine and native-image gateway from the
  selected global001. Use a fresh/invalidation-proved Rust target; preserve name160
  and ABI2. A gateway-only build is not the distributed image-engine executable.
- Build the separate managed SYCL worker counter-free and validate the SDK-hidden
  flat closure. The current unselected package has 63 unchanged vendor members,
  worker `4deb9a1b...`, manifest `fc169dd2...`, archive `85d7aecb...`; full digests
  are in READY-v1.json. Independent closure review passed under receipt
  `084d6c0a88560afc1f5ad1bf0a287834232264c6f28c87ebc29595f78f18f381`.
- Test the selected standard and managed workers with the supported generation/edit
  recipes, prefix STORE/REUSE, negative/CFG paths, previews, retained weights and
  per-request content/cache cleanup. Record cancellation, fresh recovery and idle
  SHUTDOWN, device faults, matched full performance and privacy checks separately.
  Exact synthetic matches and short routing counts alone do not establish these.
- Run normal release validation/notices/packaging/signature verification under
  separate release authorization. `package-release.sh` explicitly refuses the
  private SYCL-linked worker as the standard executable and checks forbidden Intel
  dependencies/RPATH. Do not replace that check with the private package override.

There is no API/UI or workflow contract delta. No selection, rebuild, runtime,
GPU, service, installation or publication action was performed for this handoff.

## Authorized source/package selection checkpoint

After the historical checklist above, the user authorized installing the tested
runtime and preparing release readiness when checks pass. The owning task
reports counter-free full B70 and whole cancellation/recovery acceptance. The
isolated public selection commit updates only package001/003 PINS, adds a normal
managed SYCL build entry with explicit whole ON, and documents the remaining
fresh-build/standard-gateway release gates. Source CMake defaults and all kernel,
Rust/API/UI and other-backend build settings remain unchanged.

The original helper and private packager provenance are preserved before their
historical meaning is superseded. The normal selected helper packages the frozen
manifest directly, without pin override, into a fresh archive byte-identical to
`85d7aecb...`. No compilation, GPU, service, installation or release action was
performed by this source-selection task. See
`apps/kuno/scripts/README-managed-sycl-build.md` and the immutable public-selection
receipts for commands, parity limits and the missing reviewed MKL configuration.
