# Regular expert cohorts: complete source bundle, acceptance pending

The complete regular-cohort source bundle is composed, including public
original-reference probes and distinct native completion/counter roles. It has
NOT yet compiled or run on a GPU. No speed result exists. Read the completed
[HOST blob result](kuno-prefill-host-expert-blobs.md) first. Preserve the ENTIRE
accepted HOST/granular bundle and every previous selectable variant. Target
remains Strata CLEAN first/repeat5.5006/5.3515s with the exact current public
model/context/fixture. No API, Vashti, install, release, driver or reset change.

## Confirmed source lead and limits

Current normal repeated expert interval is5.225996740s versus attention
2.420706493s. Sampled macro matrix family19.271555ms is the largest individual
matrix family (42.923519ms all matrices). These scopes overlap and must NOT be
summed/scaled into normal wall time.

`strata_prompt_expert_cohorts.rs` independently limits `validate_words`,
selection and `enqueue_small` to1..31 members. Regular32..255-member experts
still receive individual GU/DOWN launches. The accepted granular GU example
for64 members gives80 workgroups/160 SG16. Hardware occupancy is not measured;
grouping several experts could increase independent work without reducing the
current four-C reuse or doubling input-B reads. This is a hypothesis.

Existing cohort code reserves20MiB X,5MiB H and eight complete FP16 expert
images (75MiB). Its actual packed-row admission limit is4096 and raw slot
inventory is eight unique slots. Eight groups padded to256 would require at
most2048 packed rows. This is a capacity lead, NOT authority to widen a
producer/consumer or proof of every element type, view, stride and lifetime.
Audit those actual owners and vector layouts before finalizing the design.

Pinned MIT Strata's active Intel path calls `Gemm::f16` at
`sycl/src/prefill/prefill.cpp:3980-3983`; `gemm.dp.cpp:773-807` takes the
oneMKL/DPCT path. Its opaque internal tile is unknown. Do NOT claim this new
grouped policy exists in Strata or port inactive CUDA/HIP MMQ/WMMA knobs.
Kuno's actual `strata_graph` ALREADY retains a genuine dedicated copy queue;
do not falsely describe this as adding a missing transfer queue.

## Compose the whole new policy before model timing

Use an explicit new full variant with distinct authentic INFO/kernel/native
roles. Keep original1..31-member cohort grammar and admission limits intact.
Never implement this as simply raising its32-member boundary.

- Audit real routed group ordering and currently eligible contiguous regular
  groups; select bounded batches with actual same-bank raw slots. Preserve
  resident/streamed effective-source classification and both original banks.
- Price/prove every real X/H view and complete FP16 slot, packed/padded prefix,
  original map/status/fault buffer, all simultaneously live raw/dequant work
  and metadata. Do not increase original reservations/caps or borrow a live
  owner twice. Partial batches and independently varying counts are required.
- Compose real gather and zero padding, eight-way original-weight dequant,
  genuine batched GU, validating activation and ordered DOWN/status consumers.
  Retain ascending per-cell arithmetic, raw GU/pending2/fault semantics and
  original eight-member publication. Avoid unproved out-of-range speculative
  reads. Groups>=256 retain their accepted large path; original tile8 remains.
- Choose actual GU/DOWN dispatch geometry only after inspecting maximum-grid
  waste and each group's member count. Fixed maximum grids must not be credited
  as useful work. A variable packed grid needs its own genuine metadata proof.
- Compose exact native names, real grids, complete owner/phase grammar,
  initialization and completion/retirement across both banks. No fabricated
  prompt/batch scope, cached valid bits or command-limit expansion.
- Add honest actual new groups/batches/launches/workgroups/source-copy counters
  and explicit helper branches. Do not credit an old per-expert producer for a
  new batched kernel. Preserve all logical member/group/FLOP/source-byte facts.

Only after this WHOLE producer/consumer/storage/grammar policy is composed:
compile, run actual resource/SSA/layout/guard and mixed-count/original-reference
GPU controls, then both complete model/counter/erasure/recovery chains and the
unchanged original CLEAN gate. Preserve every old control and variant. Distinct
old/new numerical receipts need exact source/worker provenance; unchanged-source
reuse is allowed only with explicit, real evidence, never a relabeled receipt.

## Smaller alternative to review afterward

A two-C granular policy is another unimplemented lead:16 members x16 output
rows per SG16, WG32/two SGs, M16/N32 rather than current M16/N64. The same
64-member GU becomes160 workgroups/320 SGs with unchanged useful arithmetic,
less accumulator liveness but twice the input-B fragment reads per output.
Weight-A reads remain unchanged. Compose both regular and cohort16 consumers,
genuine guarded K32 carry, exact native roles/grids and counters together.
Two subgroups still require the ORIGINAL256-F32/1024B publication scratch;
do not falsely price512B. Occupancy and speed remain unproven. Prefer auditing
regular-group batching first rather than another isolated small-tile edit.

## Composed source policy (2026-10-24 UTC, unvalidated)

Feature `strata-regular-cohorts` / builder `--regular-cohorts` requires the entire
accepted HOST/granular/direct8/global-image/raw-lookahead/native-bulk bundle.
Original1..31 cohort grammar remains strict. New exact regular producers,
INFO schema, guarded matrices and native grammar admit32..255 members only;
>=256 retains original legacy units. Original two banks/eight raw slots and
20MiB X/5MiB H/80MiB working owners are unchanged. Eight images are75MiB and
at most2048 packed rows, within original4096. No extra runtime buffer owner.

Regular raw-bank units retain group ordering and greedily take up to8 adjacent
eligible groups while maximum row-block count is at most twice the minimum.
The genuine grid is rows/64 x ceil(max-members/16) x slots; individual guards
skip inactive Y tiles. Actual physical and active-tile workgroups are counted
separately; the plan bounds physical grid waste to2x active tile work. Original
four-C interleaved K32 granular products and eight-member publication remain.
Actual workgroup32/SG16/publication1024B still require compiler/resource proof.

The whole source includes gather, eight-slot original GGUF dequant, GU,
validating activation, ordered DOWN, exact six-phase scope, all original
owners/streams, unit-end retirement, honest new subset counters and capacity
feature label. Old producer identities do not receive new regular launch credit.
No API/consumer/installed service change; required Vashti changes:none.

Public probes are composed but UNEXECUTED:6 paired original gather/activation,
6 original GGUF dtype/mixed HOST/GPU/all-eight-slot dequant,12 full paired
regular-matrix cases against genuine existing per-expert granular GU/DOWN.
Counts include32/33/63/64/65/127/128/255, eight255 groups and partial3 groups,
with finite and nonfinite/invalid routes. Whole output/status/fault/source owners
and all original numerical controls remain required. GU reference packing is
scrubbed CPU memory; DOWN uses one original400MiB output owner. No additional
GPU output or increased512MiB probe budget. New metadata tests are not GPU proof.

Next freeze source, compile, run7 new plus all435 prior AppCPU controls, the
same-source feature-off435 controls, exact new native grammar/selector controls,
all original compiler/resource and fresh GPU numerical acceptance. Then both
complete model chains and the unchanged original CLEAN gate. Do not claim
Strata itself batches experts; this is our source-grounded hypothesis. Preserve
failures and whole variants without isolated speed-based component rollbacks.

Source review before live GPU execution found and fixed a genuine coupled
padding defect: inherited global producers usedPad32 for32..255, while new
packed prefixes usedPad16. Parallel Z groups could clear a following group's
real rows. Both new gather/activation now explicitly usePad16 and an actual-
source CPU control proves disjoint real/padding writes. Original old producers
are untouched. Initial3cdb3da3 compile failed inference at shader-list joining;
72933c8d fixes it. Its CPU run is preserved, superseded before GPU execution
by this padding correction; no pre-correction GPU or speed claim.

At aeeb8fff, fresh442 full/435 feature-off AppCPU and all16 new actual matrix
layouts pass; worker builds and offline437 WGSL/336 matrix layouts pass. Native
runtime18 baseline passes; full96 finds2 new-control failures (94 pass). Keep
this failed receipt. One exposes missing all-buffer/read-dependency stream
validation inside the distinct grammar (live backend already checks it); add
that validation explicitly. The other incorrectly demands a pre-known identity
for the final PART sink at its first binding. Correct it to prove established
owner identity and exact first-bound output span; local metadata cannot prove
an unstated sink ID. Actual graph ownership/charging and whole GPU/model consumer
checks remain mandatory. No GPU/math/speed execution yet. Revalidate final source.

Final e7718887 passes fresh442/435 AppCPU,18/96 runtime,9/10/10 selector
metadata,10 native metadata,437 WGSL/336 matrix layouts. Workerccc0114a.
Fresh HOST source/copy/audit passes, with clean physical recovery. Initial
whole supervisorv2 stops at a stale checker matrix_layouts320 expectation;
actual336 is old320 plus new16. Forward checkerv3 changes only that count;
all read-only preflight passes and fresh numerical execution begins.

The v3 physical run reaches the new paired-matrix control, after the new6
producer and6 original-GGUF dequant calls return, then fails in the probe's
private upload: `private write requires full owned same-stream span`. It used
`resident_util::handle`, a logical trimmed view, while uploading the whole
padded owner. Correct the probe to use `arena.handle(...).clone()` for uploads;
keep logical views for matrices and preserve the strict private-write API.
No arithmetic/kernel or production engine/scheduling change. Move the complete
new24-control family to the front of the SAME public numeric entry, preserving
every original control, so setup failures are seen early. Add compiled-public
completion labels only. No full aggregate GPU receipt or speed result exists.
V3 releases all GPU memory/admission: Intel42278912B,NVIDIA0,five faults0,
all3HTTP200,admission unpaused/no recovery/wait0/completed271. Preserve failed
worker/checker/artifacts. Final corrected source needs complete acceptance.
