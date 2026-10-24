# Regular expert cohorts: next coherent source plan

NOT implemented, compiled or measured. Read the completed
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
