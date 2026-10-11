# Original frozen HOST packed-word staging leaf

This source-only leaf is based on 0353369. It supplies exact copy kernels and
explicit staged consumers; it does not allocate, dispatch, change admission,
rewrite source identities, or establish GPU acceptance or a speed gain.

`grouped_moe::frozen_host_staging::Shape::new(geometry)` requires the original
frozen 512-slot, tile 8, D=2560/FF=640 profile with partition exit and generic/all-member
arithmetic. Private/workgroup tables, GU factors and shared down scales reject.
The two-row GU and four-row DOWN choices are independently allowed. Original
packed metadata, quantization, virtual64-lane accumulation and reduction order
remain unchanged. Default entrypoints pass the staging flag false.

The parent reserves, proves and erases three fixed 32 MiB PrivateArena owners:
gate, up and down. `stage_words()` returns `[8_388_608;3]`; `payload_words()` returns
the smaller format-specific 30-group payload capacities. `gate_up_words()` and
`down_words()` are the exact original expert strides. The maximum GU/down payloads
are 26,112,000/27,648,000 B, including every possible group position. Even Q2 down
receives a full 32 MiB owner. Allocation/pin/stream/alias/epoch/MAIN/INFO proofs and
full-owner close erasure belong to parent integration, with all original source
pages and pins retained separately.

`stage_gate_up(gate,up,map,stage_gate,stage_up,fault,map_first,first_slot,slot_count,
expert_words)` and `stage_down(down,map,stage_down,fault,map_first,first_slot,
slot_count,expert_words)` use 256 threads and `Shape::copy_grid(down)`, equal to
`ceil(expert_words/1024) x30 x1`. Each thread copies four stripes of words at
`cube_x*1024+lane+stripe*256`; each instruction accesses 256 consecutive u32 words
across the workgroup. The final partial stripe is explicitly guarded. No packed
word is decoded, transformed or omitted. The existing whole-model fault owner
is bound as `Array<Atomic<u32>>`; a bad binding extent ORs bit 4. The successful
copy path uses no atomics or barriers.

Both copies and consumers preserve the original shard/slot/member guard, then
also require a full frozen wave of 1..3 tokens, 1..30 groups, and one of the actual
HOST shards `(first_slot,slot_count)=(144,144),(288,144),(432,80)`. Original source
bindings have exactly `slot_count*expert_words` words. Stage bindings may expose
the logical 30-group payload or the full owner, bounded by those two extents.
Source copy offsets are `(MAP.slot-first_slot)*expert_words+word`; scratch offsets
are `group*expert_words+word`. MAP, slots, members, ranks and probabilities do not
change. Inactive groups/tails are never consumed, and malformed MAP remains
subject to the original full deferred route and finite/coverage acceptance.

The staged consumer names are `grouped_gate_up_frozen_staged`,
`grouped_gate_up_frozen_staged_two_rows`, `grouped_down_frozen_staged`, and
`grouped_down_frozen_staged_four_rows`. Their signatures match their corresponding
original frozen entries; the parent substitutes only staged packed weight
bindings and keeps the original HOST shard arguments. The original two VRAM
shards use their existing source bindings and consumers. Consumer grids and
64-thread CubeDim follow the selected original row geometry. Shared reduction
scratch remains 4096 B GU and 2048 B DOWN; staging adds no consumer barrier.

The parent must seal same-client/stream ordering: router -> host copies -> matching
consumers -> later overwrite. There are six added copy dispatches per layer,
retaining original deferred GPU router checks and original complete target
verification/repair. No CPU route fence, readiness buffer or persistent private
cache is introduced. The parent reports completed actual policy, copy dispatches
and logical copied bytes derived from accepted unique HOST groups. Copied bytes
are not measured PCIe traffic; staging changes access locality rather than the
compulsory packed source volume. Physical acceptance and same-artifact language
timing remain required before making a speed claim.

Five focused offline tests pass across the initial four-pass run and the repaired
copy-shader rerun. They cover all 48 public profiles, all three HOST shards,
nonaligned tails and inactive gaps, malformed headers, original C packed bytes
and decoded weight bits, exact original 64-lane trees and their 32/16 physical
lane bridges, and actual Checked/Unchecked WGSL plus SPIR-V with zero-init None.
The unchecked copy AST checks role, expert stride, shared intra-expert word and
tail guard; source slot versus destination group additionally relies on CPU
coverage and source review. Consumer float operations/shared extents match their
corresponding original entrypoints for every actual quantization family.

The first compile needed an explicit `8_usize` CubeCL literal. The first copy
reflection check omitted INFO: actual GU/DOWN storage bindings are 7/5, including
INFO. Both were repaired and the focused tests passed. Strict Clippy then found
four routine lints, now repaired. Root reran all5focused proofs after those edits
and strict pure-direct all-target Clippy after parent integration; both pass.
All leaf CPU jobs ended
at 13:30:48 UTC, before the 13:30:55 freeze. No GPU, services or installed runtime
were used here.
