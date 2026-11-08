# Fixed public Intel MTP placement comparison

The public original CPU13 cold/warm and compiled code/prose200 controllers
accept `KUNO_LAB_MTP_DEVICE=intel-public` as an explicit comparison. Absence
retains the Nvidia draft selection. There is no automatic GPU, profile, expert
capacity or allocation-cap fallback. Ordinary inference is unchanged.

Initial Intel placement requires the original frozen host sources and GPU
router, inactive partition exit, direct fill, inline direct-device-only build,
Verify3-fused, chunk I/O and request reuse. CPU13 additionally requires its
existing single warm replay and original diagnostic placement. Fused MTP sweeps
remain explicitly selected. Q8 is the absent-profile default; the existing
`KUNO_LAB_MTP_PROFILE=strata-q2` is a separate explicit alternative. Full draft
vocabulary, original packed weights, target reference, conditioning, accepted
prefix and real repair checks remain unchanged. English shortlist, tables,
shared DOWN/GU factors, two-row GU, staging, three members, prepared experts
and live reduction are excluded initially. Existing bounded GU integer/stride,
four-row DOWN, bounded ThreeGdn and chronological QSA may compose without new
owning charges. Existing Q8 broadcast and shared-scale head policies retain
their original profile restrictions; integer GU excludes the broadcast head.

Target-only mode rejects explicit MTP device selection or Intel memory fields.
Default Nvidia mode rejects stray Intel memory metadata. Duplicates, malformed
device names, incomplete or malformed numeric metadata reject during parsing.

Before either target or draft model allocation, the protected runner must
collect fresh Xe MEM_REGIONS data for the unique Intel discrete target while
other inference is paused and supply these canonical unsigned decimal fields:

- `KUNO_PUBLIC_MTP_INTEL_TOTAL_BYTES`: physical local VRAM total.
- `KUNO_PUBLIC_MTP_INTEL_USED_BYTES`: actual allocated local VRAM.
- `KUNO_PUBLIC_MTP_INTEL_DEVICE_ID`: the query GPU's PCI device ID.

Used must be positive and below total; privilege-masked zero never authorizes
an empty card. Source checks the query's device ID against the exact parent
target `wgpu::Device::adapter_info()`, requires discrete Vulkan Intel and a
fresh target runtime with zero main allocations. It derives full144 bank
charges and original ten-rank source page bounds from the actual bound graph
before model construction. The full original geometry still requires 144;
smaller low-bit slots are not currently supported by the source owner and
request contracts, and this comparison does not introduce them.

The combined physical admission includes every full owning bound once:

| Component | Bytes |
| --- | ---: |
| Target 144 bank pages | 18,723,373,056 |
| Target full static-cache cap | 8,589,934,592 |
| Target full private MAIN, plain / integer GU | 828,866,560 / 828,932,096 |
| Target public vectors | 25,165,824 |
| Target upload I/O | 67,108,864 |
| Target runtime I/O | 134,217,728 |
| Maximum original ten-rank source pages | 31,457,280 |
| Q8 / Q2 full draft public owning pages | 5,011,668,992 / 2,539,257,856 |
| Draft private cap | 16,777,216 |
| Draft runtime I/O | 67,108,864 |
| Draft source cap | 134,217,728 |
| Shared driver/opaque allocation margin | 536,870,912 |

With integer GU, the total is 34,166,833,152B for Q8 or 31,694,422,016B for Q2.
Plain GU subtracts exactly 65,536B. All policy-specific runtime/INFO increments
remain inside the original separately enforced 128MiB target runtime bound;
they grant no MAIN credit. The existing target 31GiB and draft 6GiB per-context
admission ceilings are unchanged. Combined admission uses the full original
owning bounds rather than adding both coarse context ceilings. It rejects
when actual available local VRAM is less than the combined charge. Root must
ensure query freshness and unchanged physical load through construction.

The draft selects exactly one Intel discrete adapter and must match the
parent target's complete actual adapter identity, including PCI bus ID when
reported. It requests a new Wgpu device/queue on that physical GPU. When a
parent target device is retained, it borrows both Vulkan HAL devices and
requires two present, nonnull, different `VkDevice` handles while both guards
are alive. No raw API calls, handle logging, retained raw handles or destruction
occur. Wgpu 29 core device equality uses context-local IDs and cannot establish
identity across independent Instances; that equality is deliberately unused.
The draft calls `init_device` again with the original custom exclusive
allocator. The independently created ComputeClient must have a different utilities/stream identity from the target.
Target request I/O is never configured on the draft; the draft retains its own
64MiB transient I/O, retirement flag and private arena. The original MTP fresh
client check, all 42 private-owner audits, public-page checks and complete close
are retained. Moving placement changes no kernel or reference arithmetic.

Receipts separate `mtp_device_requested` from actual `mtp_device` vendor,
device, PCI bus ID, shared physical GPU, distinct Wgpu device/runtime and draft
runtime ID evidence. Legacy non-frozen default scopes without a parent Wgpu
device report unknown physical target fields and `distinct_wgpu_device` as null
with `logical_device_identity=unverified-no-parent`; actual runtime independence
and separate creation are still mandatory. A retained target always requires
verified distinct Vulkan handles, and Intel always requires a retained target.
Verified trials report `logical_device_identity=borrowed-vulkan-device-handles`.
`intel_mtp_admission` records the fresh physical memory,
every combined charge and remaining capacity beyond the driver margin. No
prompts, token values, vectors or per-expert routing history are added.

Focused parser, exact charge, insufficient/masked-memory and actual identity
tests cover both controllers. CPU13 reference, real target repair, MTP numeric
parity and matched Intel/Nvidia/target-only throughput remain root-owned live
acceptance gates. No physical placement or performance acceptance is claimed
by source tests alone. No API, service, installed engine or release changes
are included.

Initial controller source validation: all 26 language-controller tests, all 21
CPU13-controller tests and all 15 target-build helper tests pass. The five Intel
selection/admission/identity tests were rerun after the final receipt and exact
comparison-policy assertions. Strict all-target Clippy with `direct-device`,
`-j2` and `-D warnings`, formatting and diff whitespace checks pass. No GPU,
service, root command or live memory query was run by this controller task.

The first Intel physical attempt passed combined memory admission but stopped
before model allocation because the earlier cross-Instance Wgpu-ID comparison
reported a false identity match. The Vulkan handle comparison corrects that
source check. Source tests cover absent/null/equal/distinct typed handles and
strict verified versus unverified receipt scopes; live placement acceptance
remains a separate root-owned gate.
