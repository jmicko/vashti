# Lane-local eight-value expert dequantization

This optional internal `strata-dequant-direct8` component follows the active
Intel native FP16 prompt path in pinned MIT Strata
`fb58e0dbc8399662c0e47c76578c6e878b14f6cf`:
`sycl/src/kernels/cuda/iq_kernels.dp.cpp` `store_run`, `dq_iq3_xxs`,
`dq_iq3_s`, `dq_iq2_s`, `dq_iq4_xs`, `dq_iq4_nl`, `dq_q2_0`, and
`iq_dequant_gu_f16`/`iq_dequant_f16`; the latter are actually called by
`sycl/src/prefill/prefill.cpp`'s native FP16 expert schedule.

Each invocation loads only its own eight-value payload and the relevant
original scale/sign fields, then publishes one original aligned vector8.
There is no sparse leader, whole encoded-record broadcast, subgroup shuffle,
SLM, barrier, extra allocation, or CPU work. The workgroup has32 invocations,
covering256 decoded values. Source, metadata, native identity, all regular,
resident, role-view, cohort and numeric launch grids change together to
`ceil(rows*columns/8/32)`. All row/quarter arithmetic uses the actual global
index, preserving640-column DOWN rows and partial final workgroups.

The original full-source descriptors, HOST/resident role routing, fixed
MAP/address/format checks, entire-owner and exact role-view extents, raw-slot
release boundaries, original six-stage ordering, FP32 factor association,
signed packed-code helpers, FP16 rounding and `store_original8` finite/overflow
fault/zero rules remain. IQ2_S retains its separate16-value scales; Q2_0
retains the selected32-value half of its encoded64-value block. Original
record32 and quad32 source/math/control populations remain explicit; original
offline source fixtures retain their original256-thread metadata.

New genuine regular identities are `StrataPromptExpertF16Direct8Image`,
`StrataPromptExpertResidentGuDirect8Image` and
`StrataPromptExpertResidentDownDirect8Image`. Cohorts use
`StrataExpertCohortGuDirect8Image`/`StrataExpertCohortDownDirect8Image`.
Runtime admission is exact and feature gated by `kuno-dequant-direct8`;
regular direct identities require the actual fixed role/shape and12800/6400
WG32 grids. Cohorts bind the real third grid dimension, immutable INFO,
all shared/private owners, original role/release rules, and a consistent
original/quad/direct producer family throughout the pair.

The production counter `expert_dequant_direct8_records` counts the same
logical32-value records as `expert_dequant_record32_records`; selected
direct8 does not increment `expert_dequant_quad32_records`. The pre-existing
96 dequant plus6 multi-cohort public GPU records expose `actual_direct8`,
`values_per_invocation` and `workgroup_invocations`; their original independent
C/Rust numerical, whole-owner, physical-source and fault checks are unchanged.
These runs exercise the new selected producer, not the old quad producer.
Old quad GPU evidence remains the prior accepted54162b57/630c4898 run.

Additive controls are4 application CPU tests,5 runtime CPU tests, and30
actual physical-source SPIR-V validations (24 regular/resident/role-view and6
cohort). The shader validator requires actual SPIR-V LocalSize32x1x1,
no synchronization/collective opcodes and standalone Vulkan1.3 validation.
The independent pinned-C fixture test covers all six formats and both FP32
values and FP16 round points; geometry controls cover final partial groups.
The previous30 quad shader controls explicitly build the retained quad
source even when direct8 is selected.

This source component was reviewed and formatted in its isolated worktree.
No build, CPU execution, GPU validation or timing was performed there. The
parent will compose it with the genuine32KiB shared-lifetime expert kernel
before complete acceptance and performance measurement. No speed claim is
made from source review or isolated old/new kernel timing.
