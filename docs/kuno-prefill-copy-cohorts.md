# Coupled prefill copy cohorts and segment compaction

Backend-only candidate, not installed or accepted yet. Keep original arithmetic,
eight raw slots/32MiB root, miss-only ownership, current source checks, full
physical views, all hazard/flight/erasure/capacity checks and original controls.
No API/model/config/frontend change. Retain every preceding component.

## Evidence and selected change

Exact public prompt-only flat CPU sample (4836abf2/SHA025a2738) attributes16.61%
to libc memset. A fixed-public-only caller capture shows18.07%, with two dominant
call chains entirely in the installed Intel Vulkan DSO. Disassembly corroborates
the caller at relative0x744c0 writing0xf538 bytes of internal state. Internal
symbol names are stripped; command-state reset attribution is an inference from
disassembly and the call paths, not a resolved driver source symbol. Original
warm expert receipts encode37959 segments and reencode40107 commands. Raw public
callchain samples were confined to protected root-only tmpfs and removed; only
static symbol/counter reports remain. No private request was admitted.

Implement BOTH changes together. The existing eight-slot miss ring initially
fills8, then refills4 only after four original consumers have been enqueued.
Consumption is NEVER GPU completion: original actual range/timeline hazards
still prevent any physical overwrite. Resident groups consume no slot and may
advance without a copy. The same cyclic miss-rank slot assignment survives.
No larger scratch, GPU allocation or model-cache reservation is introduced.

After building the COMPLETE original queue/hazard plan and retained footprint,
merge only adjacent nonheader segments on the SAME queue when the left signal
is not explicitly referenced by any peer wait and every right wait is already
covered by the left waits. This removes unobserved release-only cuts inside a
copy cohort while preserving each real early-release point used by copy work.
Retain header0, every actual node and original queue/order. Remap node segments,
wait values and final copy/shared signals monotonically; no wait may target its
own/later segment. Never infer completion or change a buffer access declaration.
A fixed4KiB reference bitmap uses the precharged worker stack; no heap map or
larger NATIVE_QUEUE_HOST_BYTES/graph/command reservation is added. A volatile-word RAII guard wipes its entire span on success/error/unwind. It excludes
header and terminal signals, which cannot be merged as left endpoints.

## Required evidence

Keep original eager-ring tests on their original policy. New mixed/adversarial
cohort tests must prove every original group/slot/copy and no live-slot reuse,
including resident runs, tails, error/finish and full-capacity metadata erasure.
Use actual Planner/Plan CPU controls plus an independent node-level queue/wait
reachability oracle, with mixed aliases/copies/shared lanes/header/early releases.
Require identical original retained byte footprint and no additional dependency
or lost real dependency, bounded stack, idempotence and capacity/failure checks.

Release/offline and original189 GPU controls plus both full73/4245/4245 modes
must pass. Real public counters must prove cohorts and compaction were exercised,
while original copy/FMA/source/owner/erasure/recovery populations remain intact.
Compare whole first/repeat, then original terminal/clean/staged/install gates if
promising. No speed claim from CPU preparation or isolated timing. Do not reset
or reboot either GPU. Installed worker3fffc266/gateway36626d30 stays unchanged.

## Frozen CPU/release checkpoint (2026-10-21)

Source661ddad3e8c3d7673f61acd2e1984bbbc9f6b591, worker SHA256
09f396123ec441a96e1c948bcf3f8f36e37a76583129ed0bc27f688c4e77d511,
build `strata-parity-copy-cohorts-20261021`. Both feature halves are selected;
optional router-register kernel is NOT selected. Original shader arithmetic and
421 WGSL/216 matrix layout inventories remain unchanged and pass offline.

166 selected actual App CPU controls pass (original154 retained), including the
original eager eight-slot policy, separate cohort mixed/tail/error/zeroization
controls, actual Plan/Planner tests over160 mixed96-node graphs, an independent
node-order reachability oracle, identical complete retained footprints, actual
peer-observed releases, stronger waits, malformed preflight/no mutation, maximum
32769-segment capacity, idempotence and full4KiB volatile bitmap erasure.
The11 original runtime capture/grammar controls are rerun and pass; their entire
source tree is byte-unchanged from accepted e454c594. No numerical threshold or
reference is relaxed. Requested CPU/release feature sets match exactly (Cargo
feature order is not semantic; duplicates are rejected).

Initial166 suite had165 passes and one unchanged BF16 shader-validator failure:
the system `/usr/bin/spirv-val`2025.1 rejects capability5116. Its diagnostic and
failed suite are preserved. Repeating with the ORIGINAL pinned validator2026.1
fbe4f3a (SHA5878b623d8855fdabdd7074349cb9aa734f06ac32bf63617a12df3afbea628fe)
passes all166; no source/test change or validator-rule relaxation was used.

Evidence under `/media/jmicko/faststore/kuno-development`:
`native-copy-cohorts-{cpu-selection,cpu-features,runtime-cpu,cpu-pinned-final,offline,release-build}-20261021*`.
The source-pinned protected wrapper `check-native-copy-cohorts-connected-20261021.py`
keeps original189 GPU checks and both full count/pool/source/owner/recovery chains,
then requires real cohort counts and actual original/compacted segment counts.
Root owns unit `kuno-copy-cohorts-20261021.service`; GPU acceptance and speed are
PENDING. No new installed selection, API/frontend/config/model/release or reset.
