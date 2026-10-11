# Source-only original four-row DOWN stride-four schedule

This distinct, default-off laboratory candidate changes the workgroup schedule
of the original frozen four-row DOWN projection. It adds no inference API,
production configuration, model format, image behavior or persistence. Source
tests are not physical GPU correctness or throughput acceptance.

## Leaf contract

`grouped_down_frozen_four_rows_stride_four` keeps the original entrypoint's five
array bindings, eight runtime scalar arguments and compile-time projection
arguments. The parent dispatches exactly `(640, 4, 1)` with 64 threads for each
of the five original source partitions. Only the authenticated hidden2560,
FF640, tile8, frozen512 IQ4_NL/Q2_0 profile is supported. This entry is separate
from the original `(640, 30, 1)` four-row DOWN entry.

Each workgroup first scans `g = group.y; g < 30; g += 4` using only the original
private MAP and public source partition. The scan and exit are uniform across
all threads. At least one valid group must exist before source payload reads or
shared initialization. The full-wave count is 1..3, group count is 1..10*count,
every selected slot must belong to its original partition, lengths are 1..count,
and all original route bounds remain required. Invalid groups receive no
readiness publication; original router, status and fault acceptance still
reject a missing or invalid route.

The second uniform loop visits the same groups and invokes the original
four-row arithmetic body with an explicit group index. Existing four-row
entrypoints pass their original `group.y` to that same body, preserving their
signatures and scheduling. Every active iteration resets all 32 private F32
accumulators per thread and all 512 shared F32 cells. The original virtual64
column order, IQ4_NL/Q2_0 weight decoding and factors, eight member positions,
`(partial0 + partial2) + (partial1 + partial3)` fold, shared strides8/4/2/1,
partials, rank order and status writes are unchanged. A final uniform barrier
protects every shared read and output before the next group resets that owner.
No workgroup zero initialization is assumed.

The single shared reduction owner remains 2,048 bytes. There are no new MAIN,
HOST or INFO owners/bindings, and no storage allocation or transfer. The full
public five-partition DOWN envelope falls from 96,000 to 12,800 workgroups per
layer. Active arithmetic and original source-weight traffic are unchanged;
this grid ratio does not predict a throughput gain.

## Immutable parent selection and completed execution

`Sources::down_stride_four` is a separate immutable scheduling boolean. The old
`new_with_policy_and_gu_schedule`, `PreparedPlan::attach_with_gu_schedule` and
`ResidentModel::attach_public_host_sources_with_gu_schedule` APIs remain wrappers
passing DOWN stride false. Their new `...with_schedules` variants append
`down_stride_four` after the existing `gu_stride_four` argument and before
cancellation, where applicable.

Selection requires four-row DOWN, inactive partition exit, the real frozen GPU
router, target rows, original Verify3 dense/fused and token checkpoint paths,
deferred token I/O, tile8 and full MAX8 owning spans. Private/workgroup tables,
shared DOWN scales, shared GU factors, packed HOST staging, three-member
arithmetic and paired GU are rejected for this initial candidate. Original
plain GU is allowed without a GU integer owner; integer GU and GU stride-four
compose through their unchanged independent policy and owner checks. No
automatic math, geometry or schedule fallback is allowed. The source geometry
continues reporting its original public grid and scratch charges.

Each real layer uses the same five native source handles, physical roots,
source identity, immutable expert/slot mapping, arena stream, MAP/activation/
partial/status owners and exact original alias/span/epoch proofs. The ordinary
CPU-router source path cannot dispatch the selected schedule. The selected
runtime requires the full public30-group envelope and count1..3, and requires
the resolved DOWN kernel to be original four-row with no staged shard.

A private `DownStrideFourDispatch` witness records the actual layer, count and
five successful distinct DOWN kernel enqueues. It is joined to the source
`Proof` and real router owners before the existing finite, fold, reduce and
all-family acceptance steps. Only the original accepted layer publication adds
`down_stride_four_layers` and `down_stride_four_dispatches`. The completed getter
`native_frozen_source_down_stride_four` first requires the healthy whole-native
completion boundary, then requires positive actual layer credit, exactly five
DOWN stride dispatches and ten total source dispatches per accepted layer.
The false policy requires zero dedicated credit. Requested flags and ordinary
source/DOWN dispatches cannot establish selected execution.

## Required verification

Focused CPU coverage in `down_stride_four_tests.rs` expands the actual entry
through both checked and unchecked WGSL/Naga/SPIR-V paths for both DOWN formats.
It checks 64 threads, exactly one 512-cell shared owner, explicit initialization,
loads after barriers, a final reuse fence, and inactive exit before shared
writes. It covers every public count/group and the original five partitions,
plus all48 original layer geometries. Parent tests exhaust the competing-policy
matrix, independent plain/integer GU policy, exact count/layer/five-call witness,
joined positive completion credit, missing/extra credit and arithmetic overflow.
Existing four-row tests retain the independent original C weight fixtures and
bitwise original64 versus four-row dot comparison.

Root must still add the explicit controller selector and receipts, freeze a new
artifact, pass the unchanged original CPU13 target/carry/greedy/real-repair gate,
complete full owner erasure and physical GPU memory recovery, then measure
matched normal language throughput with all CPU builders idle. No numerical
tolerance is widened and no speed gain is claimed by this source handoff.
