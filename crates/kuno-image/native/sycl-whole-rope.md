# Experimental whole interleaved RoPE

This default-off, SYCL-only prototype targets patch001's shared prepared-PE
engine. It changes no API, parameters, content cache, installed configuration,
or accepted release pins. GPU correctness and full-workflow acceptance remain
pending. The original accepted patch001 prepares PE inside each RoPE call and
does not satisfy this closed-window matcher; no routing claim applies to that
engine. The independent REPEAT bypass remains disabled for this experiment.

`KUNO_SYCL_WHOLE_ROPE_BYPASS` enables the private backend matcher/kernel.
`KUNO_SYCL_WHOLE_ROPE_TEST` additionally exposes oracle-only controls and nine
fixed counters. A production candidate must omit the test definition. Graph
capture builds always follow the legacy path.

The exact interleaved chain starts with F32 X `[128,32,L,1]`, permutes it to
`[128,L,32,1]`, materializes CONT, reshapes to `[2,64,L,32]`, permutes to
`[64,L,32,2]`, and materializes the second CONT. Two half views/reshapes feed
REPEAT `[2,64,L,32]`; each is multiplied as src0 by its prepared-PE component
view `[2,64,L,1]` as src1. ADD_INPLACE consumes MUL0 then MUL1, and a final
reshape exposes `[128,L,32,1]`. Prepared PE is `[2,64,L,2]` and must already be
ready before the window. Its genuine named cut is retained; inherited view
names are not cuts.

The policy proves 17 unique metadata/compute nodes and an exact seven-compute
window: CONT, CONT, REPEAT, MUL, REPEAT, MUL, ADD. Every private edge, physical
view root, offset, shape, stride, graph position and use is checked. Output/cut
flags, extra consumers, intermediate view escapes and unrelated work decline.
Final metadata descendants may share the physical MUL0 owner only through the
proved final-output chain. Because GGML propagates a final view's OUTPUT flag
to MUL0, protecting that view also declines. A downstream materialized consumer
can be the graph output while the matched intermediates remain internal.

Plans are collected before submission, limited to 64 in at most 16,384 graph
nodes. Capacity overflow or overlapping windows discards all plans. Known Qwen
Image 2.1 defaults to 32 blocks with Q/K RoPE each, exactly 64 plans; the bounded
oracle tests 64 eligible and 65/128 all-decline. Weight-derived configurations
outside this boundary fall back. No capacity is inferred from a generic Qwen
configuration.

Submission occurs only at the original terminal ADD. Skipped intermediate
storage may overlap X because it is never written. The final destination must
be disjoint from X, PE and all full view-ancestor spans; raw-X owner reuse is
unsafe because the final dense token/head layout differs. Normal same-device
SYCL buffers, actual buffer containment, 128-byte-aligned pointers, F32 strides,
bounded positive L through 16,384 and view depth through eight are required.
Padded original head/token strides are preserved with size_t byte addressing.
No builder liveness edge or allocator change is included.

For pair j, token t, head h, the kernel reads X elements 2j/2j+1 through the
original head/token strides. PE indices are `2*j+r+128*t` and that index plus
`128*L`. It stores `2*j+r+128*(t+L*h)`. Two volatile F32 products preserve the
materialization boundaries and ordered ADD. A dedicated source rule uses
precise, no-fast-math, no-association and no-contraction flags. The 1D launch
uses local size 128 and `16*L` groups; cached max-work-group and 1D item limits
must both admit 128. Capability queries occur during backend setup, with no
query or allocation in graph preparation. There is no wait, cache, device
scratch, or legacy retry after uncertain submission.

One Q/K result has S = 16,384*L bytes. The old chain stores 7S logically through
two CONTs, two REPEATs, two MULs and ADD; the new kernel stores S. This is logical
tensor traffic, not a measured physical bandwidth or performance claim.

## CPU checkpoint and native acceptance

The policy/address suite passes 4,227,553 checks, 134 graph fixtures and 24
address cases under system C++ and strict Intel host compilation. The real
gallocr suite passes 2,130 checks across 64 cases, including 56 produced-X
eligible cases, 32 actual GraphCutTensorCache/SegmentGraphBindings cases and
40 skipped spans overlapping X. Its produced DUP/NORM/RMS/MUL and prior-fuser
fixtures establish bounded host allocation/address behavior only. This corpus
needed no builder liveness edge and observed no final-destination overlap;
independent policy cases require that overlap to decline. Actual device and
full-model routing remain unmeasured.

The CPU oracle passes 78 cases and 1,368 output-vector bit comparisons under
both compilers. Native SYCL adds six large cases, requiring 84 cases and 1,392
comparisons. Cases include produced X, arbitrary and public Qwen cosine/sine PE,
strided owners, special IEEE words, escape/output/noninterleaved fallbacks,
64/65/128-plan boundaries, and narrow/full-mantissa plus all 255 finite exponent
encodings. Wide finite input arithmetic may overflow or produce NaNs; every
output word is still compared exactly. Each old/new/replay sequence is repeated
with changed PE in a second concurrently live graph, proving a changed result
and a distinct rebound PE pointer. Inputs and padding remain byte-identical.
CPU arithmetic checks do not prove device NaN payload/zero/denormal behavior.

The default invocation initializes only CPU. Native invocation is
`--device SYCL <canonical-PCI> [--benchmark]`. Strict Intel host flags are
`-fno-fast-math -fp-model=strict -ffp-contract=off`. The native backend contains
only WHOLE_BYPASS/WHOLE_TEST definitions, with repeat/residual fusers off, so
forced legacy mode executes all seven original operations. Native comparison
against independent CPU tolerance is insufficient: exact old/new output words,
replay/change-PE/pointer-rebound proofs and actual positive selected/launch
counters are required. Explicit fallback cases must have zero launches.

Success prints fixed numeric case rows and
`whole_rope_completed cases=84 comparisons=1392 bitwise=1 replay=1 changed_pe=1 pointer_rebound=1`.
Mismatches print only fixed category counts: NaN bits/class, infinity, signed
zero, subnormal and other finite. Optional timing measures synchronized complete
graphs after correctness, with three warmups and five trials of four evaluations
per mode on public-PE large cases. It is synthetic and does not establish full
generation duration or image fidelity.

Backend exception, failed compute status or uncertain false/null initialization,
allocation/reserve result emits only `whole_rope_error` case/phase/cleanup integers
and immediately calls `_Exit` in device mode before host DMA buffers or native
resources unwind. There is no retry, device reset or recovery in the oracle.
Confirmed-sync numerical mismatch uses normal scoped cleanup once. Nine CPU
failure fixtures and four argument rejections pass both compilers, distinguishing
normal cleanup from fatal exception/status/false-result no-unwind witnesses.
Existing SYCL synchronization uses wait(), not wait_and_throw(); these checks
do not assert a stronger drain guarantee. GPU execution, fault monitoring,
privacy/cancellation and later merged-residual validation belong to the protected
owner harness. No worker or production package is selected by this checkpoint.
