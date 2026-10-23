# Expert-ordered prompt output and vector combine

Backend-only optional `strata-expert-ordered-output`, composed after327124a1.
This is a source contract, not a completed implementation or speed result.
Root owns integration, feature/CLI wiring, closed runtime grammar, builds and
GPU validation. Isolated producer and consumer tasks share this contract.
No public API, model/context/configuration, persistence, Vashti, install or
release change. The installed worker remains unchanged.

## Confirmed active Strata difference

Pinned MIT Strata fb58e0dbc8399662c0e47c76578c6e878b14f6cf writes FP16 expert
DOWN into contiguous expert-ordered `Dm + o0*N` (`prefill.cpp:3983`). Its default
`moe_combine4` reads through the route inverse, performs ten chronological
rank FMAs for four columns and merges the shared expert (`kernels.dp.cpp:1839–1872`).
Kuno's original wide DOWN instead stores each accumulator through SLM, barriers,
scalar finite checks and scattered token/rank PART writes. Its original scalar
rank reduction and shared merge use separate launches. Change these together;
do not time a new producer against the wrong old consumer or infer a gain from
isolated kernels. The measured expert phase includes transfer and execution;
this difference does not establish that all of that time can be removed.

## Complete layout and original authority

Retain the exact PART allocation and all original source/map/status/fault owners.
Under one closed optional feature, every actual production DOWN family writes
`(MAP[1028+group] + member)*2560 + column`, including scalar/macro/narrow,
macro64, wide prefix/tail, cohort and guarded deep variants. Original explicit
kernel definitions and public oracle entry points remain available. New
generated entry identities must identify the actual new layout; no renaming
an unrelated shader to satisfy the old validator. INFO, role, grid, view extent,
phase, source authority, lifetime and erasure checks remain.

After successful audited router readback and validated grouping, construct the
exact permutation inverse `inverse[MAP[1540+ordinal]]=ordinal`. Validate full
coverage, uniqueness and range and its inverse relation to MAP. Publish it into
the existing512KiB router RAW view, not the separate weight-DMA banks. Its last
old RAW consumer is the completed audited readback. Retain genuine upload and
finisher read bindings and all next-layer overwrite/overlay dependencies.
Use zeroizing host memory and the existing bounded group/upload envelope.
No new GPU payload allocation or retained private cache is needed. Every next
layer resets RAW before routing; full owner erasure and loan restoration remain.

## Pending direct output and finite closure

Valid complete wide DOWN fragments can store directly to global expert-ordered
PART. The original route-based DOWN status uses2 to mean written but finite
validation pending. This is never an acceptance marker. Only the genuine
column-zero producer writes that marker. Invalid routes retain checked ordered
fallback publication and original fault128. Non-direct producers preserve their
original finite-zero logic and status1/0, using only the new PART addresses.

The consumer uses one complete workgroup per token. It snapshots the ten
GU/DOWN markers, inverse indices, weights and pending routes' raw column-zero
finite flags, then performs genuine workgroup/storage synchronization before
any sanitation or status normalization. All column work uses those immutable
snapshots: no dispatch reads a marker while another workgroup changes it.
Missing marker0 never authorizes stale PART. The validated inverse is a
bijection, so different token workgroups own disjoint PART sanitation ranges.

For pending2, original nonfinite components become0 and raise128. Nonfinite
column zero invalidates the route for all columns. A nonfinite other column
does not independently invalidate that route; its sanitized0 participates in
the original reduction. Normalize DOWN to the same original1/0 only from the
captured column-zero predicate and GU status. Preserve failure poststate as
well as successful answers, including untouched invalid-route storage.

## Vector combine and shared merge

Process four columns while retaining all ten rank FMAs in their original order,
including rank0 with the original zero accumulator. Preserve finite/positive
weights, weight-sum tolerance, fault256, original output-stride/owner checks
and audit stamp. Invalid reduction leaves the original routed output unchanged.
Fused shared merge must reproduce the original separate multiplication/addition
round points, sigmoid and fault1. When reduction is invalid it must use the
retained old routed value, exactly as the original second launch would. Retain
the original routed and combined owners if needed; do not silently change that
failure behavior to avoid a binding or write.

## Acceptance

Keep all original CPU/compiler/operator/model populations and exact numerical
limits. Add actual source/compiler controls, full-owner paired old/new GPU
fixtures for all selected producer families, inverse/metadata failures,
pending/missing status and nonfinite components, plus combined consumer and
shared-merge parity. Include authentic small tails and noncontiguous routes.
Original capabilities remain authoritative: these Vulkan devices report48KiB,
so64KiB deep variants are unavailable and cannot count as executed GPU tests.
Do not change the limit or report a capability rejection as numerical coverage.

After complete source and correctness/recovery acceptance, measure both original
full-model proof chains, inspect actual selected-kernel counters and compiled
ISA, and perform a clean matched comparison against Strata. Root owns all GPU
use; no simultaneous compilation during timing. All receipts contain public
fixture identifiers and aggregate metadata only, never private prompts, images,
tokens or expert contents.
