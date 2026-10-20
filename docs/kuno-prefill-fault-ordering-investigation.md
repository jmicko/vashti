# Expert fault ordering source investigation, 2026-10-20

Read-only analysis of the selected native prompt path at integration
`3fffc266383917cada88fd176b273db8a6b737f4` and original MIT Strata
`fb58e0dbc8399662c0e47c76578c6e878b14f6cf`. No hazard model, access mask,
fault guard, source, API, service or GPU setting is changed by this note.

## What the native graph currently orders

In `vendor/cubecl-wgpu/src/compute/strata_graph.rs`, WGSL registration uses
actual Naga usage, while `register_spirv_module` marks every originally
ReadWrite binding with both SHADER_READ and SHADER_WRITE. It does not refine
those declarations into the actual selected kernel's atomic/ordinary uses.
`node_bindings` preserves physical read dependencies; `node_accesses` merges
them by actual VkBuffer. The encoder inserts a WHOLE_SIZE buffer barrier when
either its remembered or incoming access writes. `node_ranges` has actual
descriptor intervals but reduces the access kind to an ordinary `write: bool`.
The original `strata_schedule.rs::update_regions` therefore waits for previous
writers (and, for a write, readers) on every intersecting peer-queue interval.

All six nodes in one native expert group receive the same fault handle:
Gather, GU dequant, DOWN dequant, GU product, activation and DOWN product
(`strata_prompt_experts.rs:329–405`). Their source uses fault[0] only for
unused-result `atomicOr`/`fetch_or`; the matrix's ordinary status read remains
separate (`strata_matrix.rs:529–547`). Source inspection is not yet a complete
compiled-module def-use proof for every selected kernel variant.

The long `resident_model_prompt.rs::run_prompt_full` path does not enter a
NativeStrataShared QueueScope around these experts. Its shared projections
follow the expert interval on MAIN. Lookahead raw copies bind original source
and destination, not this fault buffer. Common fault accesses consequently
do not explain long-path copy/compute queue waits. MAX8's actual shared branch
already has a separate fault owner and an explicit Fold; that join is required.

## Why fault-only removal is not yet a justified large optimization

Many barriers carry a genuine workspace dependency in addition to fault:

* GU/DOWN dequant bind the same complete stage1 root, although they write
  different image ranges; the current local buffer-level barrier sees both.
* GU output is read by activation.
* Activation writes hidden data and status read by DOWN.
* The next group's Gather reuses hidden's stage0 storage after DOWN reads it.

Thus omitting one fault barrier entry often leaves the same Vulkan barrier
call and required ordering. It has not been shown to explain the remaining
expert interval, nor that removing it enables substantial independent math.
Do not confuse the existing serialized public GPU timestamp sampler with a
measurement of ordinary overlap or a clean barrier-cost experiment.

Original Strata also uses one **in-order** compute queue here:
`sycl/src/program/generate.cpp:2862` selects it; `prefill.cpp:917` adopts it as
`m.cs`; native DQ/GU/SwiGLU/DOWN all use it at `3968–3983`. Copy has a separate
in-order queue (`prefill.cpp:969`, `sycl/include/dpct/device.hpp:616–627`) and
real copied/used events. It is not an out-of-order expert pipeline silently
running every independent group at once. Its kernels also lack Kuno's same
fault/status contract, so their guard behavior must not be copied blindly.

There are real fault readers elsewhere: route finalization
(`strata_prompt_route_finalize.wgsl:24`), GDN compatibility flags
(`strata_prompt_gdn.wgsl:14`), QSA flags (`strata_prompt_qsa.wgsl:34`), and
eventual host acceptance/clear. None may observe fault before original prior
writers complete and become visible.

## Bounded possible follow-up, not implemented

First count, only in the compiled-public diagnostic, whether a proven
fault-only exception would remove a whole barrier/event or just one redundant
entry. Do not implement a new hazard mode before this establishes useful scope.

A future conservative per-binding classification would inspect actual Program
SPIR-V: original set/binding and u32 type, constant fault[0] access chains,
only unused-result OpAtomicOr, and no ordinary/atomic load, store, other atomic,
used result, escaping pointer, unsupported alias or unknown pointer flow.
Metadata/array-length uses are not payload reads but still require proof.
Preserve actual atomic scope/semantics: selected CubeCL uses Device scope with
UniformMemory and AcquireRelease; selected Naga uses Device/UniformMemory.

Keep ordinary access flags and add any proven OR-only class separately. Only
OR-only versus OR-only can potentially omit mutual ordering. Any ordinary
alias, fault reader, clear, copy, recording boundary or final host acceptance
must still join **all** real outstanding writers on the original queues and
preserve full visibility. Unknown modules fall back to existing ordering.
No numerical, ownership, queue/fence, replay or privacy requirement is waived.
