# Opt-in native32 routed experts: CPU checkpoint

The `strata-native-expert-wide` candidate connects full128-member prefixes to
native M128xN128xK32 FP16/F32 math and authentic original M32/M8/M16 tails. It is
not selected, installed or GPU-validated by this checkpoint. No API, quant,
precision, numerical tolerance, private cache or original resource owner changes.

`strata_matrix_native_expert_wide.rs` retains the accepted native32 typed vec4
staging, chronological K16 products, four accumulators per SIMD16 subgroup,
512-thread workgroup and49,152-byte shared allocation. Publication reuses the
original16KiB scratch in four phases; it never materializes a64KiB output tile.
The accepted dense module is unchanged. The original macro/narrow arithmetic
bodies change visibility only so the new tail wrappers call their real products.

Selection requires count256..4096, fixed GU2560->1280 or DOWN640->2560 geometry,
original aligned half-vector views, actual CMMA F16/F16/F32 support,512 invocations,
49,152 shared bytes, valid grids and the original Graph physical admission.
Unsupported views/devices return false before launch. All full owner bounds,
streams, grids, tail subview and fallible raw arguments are checked before either
matrix piece is emitted. A selected path never falls back after partial launch.

Both GU and DOWN validate original map/header/route bounds. GU writes original
member*1280; DOWN scatters to original route*2560 in the complete PART owner.
Nonfinite values become zero and set fault128. Exactly one feature-zero write
per DOWN member updates the original `status[ROUTES+route]`; other-feature
nonfinites retain the original status policy. Map/status/fault and GU/PART views
stay complete. No contiguous DOWN scratch or later scatter was introduced.

## Exact producer contract

Four actual generated entry identities bind the role with a literal in each
wrapper, rather than inferring an opaque comptime DOWN flag from dimensions:

- `native_expert_wide::expert_wide_gu::ExpertWideGu<WgpuRuntime>`
- `native_expert_wide::expert_wide_down::ExpertWideDown<WgpuRuntime>`
- `native_expert_wide::expert_tail_gu::ExpertTailGu<WgpuRuntime>`
- `native_expert_wide::expert_tail_down::ExpertTailDown<WgpuRuntime>`

Their six genuine bindings are `[input, weight, output, map, status, fault]`.
Seven runtime **u32** scalars are `[columns, rows, full_count, weight_first,
group, member_base, piece_members]`. Original InfoBuilder packing puts them in
u32 words0..6, zero-padding word7, then the six original physical and six logical
array lengths in words8..19:80 bytes total,10 u64s, dynamic_metadata_offset10.
The actual generated KernelDefinition has exactly six buffers and seven U32
scalars; original INFO construction/padding remains unchanged. Backend validation
must check actual generated records and binding lengths, not assume this layout.

Prefix: base0, members=`full_count/128*128`, grid(rows/128,prefix/128,1), WG512.
Tail: base=prefix, members=full_count-prefix in1..127; grid(rows/64,
ceil(tail/tile),1), WG64. Tile is8 for1..8,16 for9..31,32 for32..127.
The tail's actual input Handle preserves allocation/stream/root size while:

- start becomes old_start+prefix*columns*2;
- end becomes old_end+(old_size_in_used-full_count*columns*2);
- used length becomes exactly tail_members*columns*2.

The other five views are identical to the prefix. Local product members become
`member_base+local_member` for GU stores and route lookup, so the tail cannot
silently overwrite member0. Full map/header count remains the original count.

The coupled bulk parser/receipt extension is separately owned. It must preserve
six logical phases while recognizing a wide prefix and its required adjacent
nonoverlapping tail as one matrix phase. Report actual seven/eight physical
launches rather than pretending there were six. Do not enable this feature in
the existing strict six-physical-command collector before that complete repair.

## Completed CPU checks and remaining gates

The actual new production module plus original native product helpers typecheck
using existing CubeCL/half/WGPU/Kuno-text rlibs in a separate CPU harness. No
shared Cargo target or GPU is used. Five focused tests pass in15.09s:

- All64 actual shader modules: two roles x prefix/tail8/16/32 x eight original
  layout combinations, through real CubeCL SPIR-V lowering and `spirv-val`.
  Native32 modules retain512 threads,49,152 shared bytes, SIMD16, eight native
  products/four stores, finite/fault handling and no relaxed FPFastMathMode.
  Tails retain the actual original product shared envelopes.
- Genuine native32 and original-tail output mappings cover every member/feature
  exactly once across256/257/264/265/287/288/383/384, with one feature-zero store
  per member and ordered K16 coverage.
- Canonical tail subviews preserve real offsets/allocation/stream and reject
  changed pieces, owner lengths and streams.
- Actual generated definitions/role names plus original InfoBuilder scalar,
  padding and array metadata checks.
- Plan/threshold/device-capability negatives and complete member coverage through
  the original4096 bound.

The harness substitutes no GPU execution. It does not replace whole-worker
compilation, the new shared native26 actual routed numerical cases, all previous
operator gates, real private request erasure/rebind, injected terminal recovery,
staged thinking/tools/semantic quality, physical recovery or matched clean
performance acceptance. Original36 GPU matrix cases remain unchanged; the source
validator only appends the new64 offline routed layouts.

Artifacts: `/media/jmicko/faststore/kuno-development/prefill-native-expert-wide-cpu-20261020`.
The separate shared oracle/native26 probe is source548ac64e; the native14 CubeK
contract remains unchanged. The source-only proposal and exact histogram are in
`kuno-native-wide-expert-proposal.md`. No throughput or deployment claim yet.
