# Full arithmetic walks for partial small expert tiles

## Resumed candidate: one accumulator loop (2026-10-24 UTC)

The user has returned both GPUs. Next internal candidate is the explicit
`strata-single-k-walk` / `--single-k-walk` feature, requiring the entire
full-walk bundle. It preserves all original launch identities, ABI, owners,
resources, guards, publication and cohort code. Only the macro computation
selection changes: both full and partial tiles use the same chronological
eight-C loop. Its existing masked B helper still directly loads each wholly
real 16-row input block and stages only partial blocks. The original separate
full/partial loops remain feature-off, not rolled back or replaced globally.
The actual shader proof must now find exactly ONE eight-product K loop for
macros, with all prior CFG/reaching-definition/B-alias/handoff negatives.
This tests whether merging eight accumulator values across separate loops
caused the observed 80:80 register spills. No speed or acceptance claim yet.

## Completed checkpoint and user-requested pause (2026-10-24 UTC)

Frozen source `b2621286610dced4cc5f8dc06bc53a7edfa6f4ef`, worker
`832e35b85dc0dabba09c5b2a1084b1a346319b89827d083b1ea747506185d696`:
the complete bundle now passes 398 AppCPU, 395 feature-off AppCPU,
18 feature-off/67 full runtime, 8+9 metadata, 10 native controls, all
48 new emitted shader layouts, and all 1223 GPU controls (1103 retained,
116 new full-owner/F64 and four new packed previous/new controls).
Both original 18-script full-model/recovery chains pass, including original
capacity, ownership, finite/status, reuse, rolling and erasure proofs.
No API, installation, service configuration or Vashti changes were made.

Normal warm/first/repeated prefill: 4.949937043/13.220987042/10.299127723s.
Diagnostic: 4.938631303/13.268106496/10.402017969s. Large requests are the
same 4245-token public fixture. This improves the preceding global-small
normal repeated 11.620150934s, but remains slower than local-route 9.360428477s;
it does not replace the best CLEAN 8.394039699s or match Strata CLEAN 5.3515s.
No new CLEAN run was triggered. Retain complete variants; no isolated rollback.

The actual 203-program ISA export contains 12 old/new small programs. The
four additional macro programs have 80:80 spills:fills, while the retained
four macros and both old/new cohort pairs have zero spills/fills. Macro
function basenames overlap across namespaces: use exact hashes/compiled
program records rather than attributing results by basename alone. The
original two QSA hashes remain unchanged. Register pressure in the combined
eight-accumulator path is a concrete next investigation, not a speed claim.

CPU V4 compiled the final source and passed the three new controls, but an
existing wide-pipeline test collided with the test-only compiler export
filename across contexts. V5 scopes export to the new controls and reuses
the exact frozen V4 test binary; all 398 and preceding 395 controls pass.
This is not a duplicate-compilation claim. All failed earlier logs remain.

Artifacts under `/media/jmicko/faststore/kuno-development`:
`small-full-walk-app-cpu-v5-safe-20261024.json`, root build receipt/freeze,
`strata-parity-small-full-walk-20261024-expert-cohorts-public-summary.json`,
`small-full-walk-{profile,normal}-safe-phases-20261024.json`,
`small-full-walk-complete-isa-export-20261024.json`, complete driver log,
`small-full-walk-unit-terminal-20261024.log` and
`small-full-walk-pause-state-20261024.json`. Complete driver exits 0;
unit invocation `8d8f4790492d4a72aab005f5a13eb659` is terminal.

Independent pause check at 06:14:31 UTC: Intel 50,688,000B, NVIDIA zero
compute processes/571MiB desktop memory, all five classified kernel faults
zero, Kuno/Ollama/Vashti HTTP 200, admission inactive/unpaused, no recovery
required or waiting requests. Bounded sudo refresher PID2712660 stopped.
User requested GPUs back and goal paused; do not launch another GPU test.

Source-only next lead, not implemented: the two genuine 4096x2560 FP16 input
views are already 20,971,520B each. BOTH gather and hidden-activation producers
could explicitly initialize authentic padding for medium/partial tiles, with
matching closed native writer/reader authorization and global-only consumers.
Spare arena capacity or initialization at allocation is not valid per-reuse
padding evidence. Preserve original real counts/grids/phases/owners. Consider
this together with accumulator register pressure; do not select isolated parts.

The chronological source/CPU checkpoints below are preserved as history;
their pending statements are superseded by this completed checkpoint.

Next coupled internal bundle after source d36a393f/full1103 acceptance. No new
performance claim. Keep all earlier variants and395 AppCPU/1103 GPU controls.
No API/UI/installed configuration, precision, original capacity, ownership,
status, privacy or erasure changes. Target remains the same Strata clean run.

The completed global-small bundle isolates a regression in macro/cohort matrix
work, with unchanged transfers/dequant/host/wide paths and zero new ISA spills.
Its partial or invalid subgroup repeats a two-C K loop over feature/member
subtiles. This source difference is a candidate cause; not an isolated proof.

Compose a distinct explicit `strata-small-full-walk` feature requiring the
entire global-small bundle, plus matching build option, native propagation,
exact generated identities/closed grammar, producer counters and proofs.
Keep original macro32/M32/N64/WG32, macro64/M64/N64/WG64 and cohort16/M16/N64/
WG32 grids and original scalar/owner layouts. Tile8 and wide tails stay intact.

Both full and partial macro cells retain all eight C fragments through ONE
chronological K16 walk. Fully real input32 uses the preceding pure global
weights-A/input-B walk. Partial input32 loads each B16 from the original view
when all16 are real, otherwise writes every cell of the SAME bounded per-SG
16x16 FP16 tile with real input or zero, synchronizes, loads B into its matrix
value and synchronizes before reusing the stage for the other B16. Both B
values must be actual independent SSA matrices before overwrite; do not assume
a pointer alias still holds previous input. Reuse one A across both B values,
four A loads/eight products per K. Never read inactive original input rows.
The stage remains64 vec4/SG512B; publication scratch remains128 F32/SG512B.
No shared memory is added beyond accepted macro2048/4096B. No WG barriers.

Cohort16 already has authentic initialized16-row padded input. Its four C
fragments always use one pure global K16 walk, including partial/invalid
subgroups. Compute arithmetic independently of route validity, then retain
original raw fast stores for wholly valid real cells. Otherwise publish each
16-member fragment through the exact existing independent8-half publisher,
skipping fragments with no real member. This preserves partial-half finite/
status rules, untouched invalid sentinels, sticky faults and deferred raw GU/
ordered DOWN consumers. No aggregate fault read or weaker AtomicOr proof.

Retain previous actual shader proofs/negative mutations and all1103 cases.
Add actual compiler CFG/provenance, per-SG masked-stage full initialization,
load-before-overwrite/both B SSA identity and K chronology proofs; exact real
WG/shared/SG/barrier/owner/role checks; corresponding metadata positives and
forgeries. Add complete previous/new F64/whole-owner and genuine eight-slot
packed numerical coverage without replacing original global-small producers.
Finalize exact additive populations before live execution. Then run the same
full-model/recovery chains and actual old/new producer partitions. No speed
selection or isolated rollback before this full bundle is composed and passes.

## Source composition checkpoint

The entire new source candidate is composed: full eight/four-C arithmetic,
independent B matrix values loaded before reuse of the original512B/SG stage,
original publisher, six distinct real entries, production selectors, exact
native namespace/ABI authentication, profile classification and real counters.
New compiler controls inspect actual CFG/product populations, independent B
SSA and actual shared-store/load/barrier chronology, retain genuine original
WG/descriptor negatives and add a real B1-to-B0 alias mutation. Exact original
shared2048/4096/1024B remains required. No compilation/GPU acceptance yet.

Final additive populations: three AppCPU (masked-stage mathematical control,
actual complete compiler/negative proof and actual generated profile names),
retaining395 =>398; four genuine metadata controls retaining63 =>67 full
runtime;116 complete-owner/F64 plus4 packed previous/new GPU cases retaining
1103 =>1223. All old global-small callbacks still launch the old genuine
producers. New paired references use accepted global-small tile16/macros;
tile8 and wide tails are untouched. Original18 feature-off runtime,8+9
metadata and10 native still required. Neither full compilation nor live runs
have executed for this new bundle. Do not interpret speed until complete.

First398 CPU build at2598d134 compiles successfully, then the new actual
shader SSA proof rejects an unrecognized B value before any new GPU test.
No numerical failure or speed result is claimed. The proof remains strict;
next diagnostic records the missing SSA ID's actual defining instruction,
then determine whether it is an unsupported valid provenance or a real
producer/SSA issue. Do not simply bypass the independent-B/overwrite check.
Failed CPU log/artifacts `small-full-walk-app-cpu-20261024` are preserved.

The second compiled diagnostic identifies the actual missing definition:
`OpLoad` from Function storage, not a missing cooperative operand or evidence
of incorrect arithmetic. New proof traces exact reaching `OpStore` values on
EVERY CFG predecessor in the SAME K iteration, rejects uninitialized and
previous-iteration paths, then follows those values to actual global/shared
CMMA loads. Independent B-origin sets and the real alias negative remain.
Test binaries may export fixed compiler modules to an explicit per-run CPU
artifact directory for faster standalone proof debugging; private worker
builds omit this export entirely. These contain no request/model/GPU data.
One actual repository masked-stage mathematical test passes separately; this
is not full app/shader/GPU acceptance. V3 compile/proof is next.

V3 compiled successfully and resolved genuine Function-local matrix origins,
then rejected a chronology assertion that assumed physical SPIR-V block order.
The actual helper branch blocks are serialized after their consuming product
block; execution order comes from CFG edges. The proof now uses exact
instruction reachability inside one K iteration (excluding its backedge),
requires every store before its stage load, both subgroup handoffs, and a
handoff before EVERY write that reuses either stage in actual CFG order.
It retains independent actual B-origin sets and exact owner/WG/SG/resources.

A standalone Rust checker compiles the exact production proof-function prefix
against the SAME cached production AtomicOr classifier and the already
serialized public module; no source normalization or GPU work. First
macro32-GU/layout0 module eb5d5fb26f6cd7dcccbf84cd136e12387f1ad496c0d6c0c3712b71049de469ee
passes positive proof and genuine descriptor/WG/B-alias/missing-matrix-store/
removed-stage-handoff negatives (`full-walk-ssa-debug7-20261024`). This is ONE
module, not all48 layouts or full398 acceptance. Earlier debug attempts kept.
The metadata-only source checker makes future proof diagnostics fast without
repeated whole-worker compilation. V4 full398 compilation/acceptance is next.
