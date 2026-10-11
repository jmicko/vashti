# Frozen three-member arithmetic scratch

This source-only leaf is based on c3e9403. It adds an explicit public frozen-only
Geometry policy and six named GU/DOWN entrypoints. Parent integration owns the
runtime selector, factory, actual dispatch receipt and physical acceptance.

The actual frozen constructor, GPU router and deferred source proof already
limit full verification waves to one through three tokens and at most 30 unique
expert groups. Each group has at most one member per token. Existing arithmetic
still allocates eight member cells, predicates all eight MAC paths and reduces
all eight cells. The new policy removes only member cells 3 through 7. For each
active member, original packed weight decoding, column iteration, multiplication
and accumulation order, virtual 64-lane tree and final output remain unchanged.

`Geometry::with_three_member_arithmetic()` requires frozen sources, partition
exit, capacity 512, tile 8, D=2560/FF=640 and generic all-member arithmetic. It
rejects secondary, paired GU, private/workgroup tables, GU factor and shared down
scale policies in both construction orders. Four-row DOWN composes independently.
`three_member_arithmetic()` reports the construction-fixed policy;
`arithmetic_members()` returns 3 for it and the original tile otherwise.

Tile 8 still defines `first_member`, MAP offsets, grid Z, full MAX8 input/output
owners and MAX80 status. MAP stays 804 words. All owner reservations, packed
source bindings and public grids stay unchanged. Consumers add a workgroup-uniform
full-wave guard for count 1..3, wave_first=0, wave_count=count, groups 1..30 and
first_member=0, together with the original shard/member guard before any shared
initialization or packed load. Geometry never derives a pipeline from a private
runtime count or group length.

| Kernel | Local F32 accumulators per thread | Reduction shared bytes |
| --- | ---: | ---: |
| Plain GU | 16 -> 6 | 4096 -> 1536 |
| Plain DOWN | 8 -> 3 | 2048 -> 768 |
| Four-row DOWN | 32 -> 12 | 2048 -> 768 |

Each retained member still initializes every original virtual-lane partial and
every shared cell before its first barrier. The four-row DOWN retains original
stride-32 then stride-16 register folds and shared strides 8 through 1. Plain
trees retain all original strides 32 through 1. Barrier count does not change.
Per-member reduction additions remain 63; removing five unused members reduces
member-cell shared traffic and reductions by 62.5%. This is a source count, not
measured occupancy, spill, kernel latency or whole-language throughput.

The original-signature entrypoints are:

- `grouped_gate_up_frozen_three_members`
- `grouped_gate_up_frozen_staged_three_members`
- `grouped_down_frozen_three_members`
- `grouped_down_frozen_four_rows_three_members`
- `grouped_down_frozen_staged_three_members`
- `grouped_down_frozen_staged_four_rows_three_members`

Staged counterparts keep the exact original HOST guard and group-position packed
scratch addressing; copy kernels, three fixed 32 MiB owners and same-stream
ordering remain unchanged. Original VRAM source shards use the plain counterparts.
All existing wrappers pass their original tile as arithmetic capacity, so default
kernels and all previous experimental entries keep their existing behavior.

Packed read volume and active MAC count do not fall. The possible gain comes from
fewer local accumulators, shared cells and unused member branches, particularly
for four-row DOWN. Physical drivers can optimize declarations or dominate cost
elsewhere, so a same-artifact language trial remains necessary. This does not
relax original complete target logits/carries, greedy repair, route/finite proof,
owner erasure or recovered-host gates. No GPU, services or installed runtime were
used for this leaf.

The three focused offline tests passed: all 48 Geometry profiles and policy
composition; original C packed-weight bits and SIMT F32 trees for counts 1..3;
and all six actual Checked/Unchecked shader entries with SPIR-V workgroup zero
initialization set to None. The shader proof checks 192-cell initialization,
shared and local array extents, unchanged bindings/barriers and original floating
instructions. Its reducer comparison sorts only hoisted uninitialized variable
declarations because the compiler emits those declarations in HashMap order.

Strict production Clippy passed. The final main-and-test Clippy run found one
test-only collapsible-match lint; its syntax repair is saved here, with the rerun
held for the parent's timed GPU trial. A declaration-only assertion was also
added after the passing shader run. Both final test-only changes still need the
focused rerun and strict test Clippy before parent acceptance; production source
is exactly the source exercised by the three passing tests.
