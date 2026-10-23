# Coupled HC down and injection projection

Backend-only optional source continuation from8b1fd224. All preceding complete
experimental features remain. This document describes source, not completed
build/GPU acceptance or a speed result. No installation/release is authorized.

The fixed-public six-phase4096-token/layer24 diagnostic reports approximately
norm1.503ms, down1.205ms, injection0.990ms, low0.422ms, up1.363ms, mix1.169ms.
These are individually completion-fenced wall intervals, not pure GPU times.
Do not compare their sum directly with clean timings or native timestamps.
Pinned MIT Stratafb58e0db's "hc read"185ms also excludes most fused residual
write/normalization: that runs at the preceding half's end before the next
`pt.mark(kPfHc)` with `normed=true`. Kuno's GR interval includes that work.
Strata's Intel default uses oneMKL via `sycl/src/prefill/gemm.dp.cpp:762`.

There is a grounded combined-projection reference in pinned Strata
`src/prefill/wmma_gemm.cu:480-574`: two authentic BF16 weight pointers form
[down;inject], one input image feeds both, and the epilogue splits two real
outputs. Its `STRATA_PF_HCDOWN` implementation is AMD gfx11-only and opt-in;
it changes K-order and global input padding and is quality-gated. This Rust
implementation copies the fusion concept, while retaining Kuno's original
Intel8x16x16 chronological K16 products and current pitch40/K32/register-stage
strategy. It does not claim those AMD options are active in the Intel baseline.

## Complete coupled source

Feature `strata-hc-down-inject` requires the entire current optional bundle,
including register normalization, dense prefix, pitch40/direct publication,
register staging, native-wide32 and real subgroup publication. Parent task owns
Cargo/CLI feature wiring and final coordination integration.

The original down projection already computes three128-column tiles:320 real
columns and64 discarded zero columns. The new producer places the four actual
injection weight rows at320..323, leaving324..383 locally zero. Both initial
stage and next-register loads use the ORIGINAL distinct down and injection
BF16 descriptors with exact payload lengths. No concatenated GPU payload,
expanded source view, source mutation, allocation, or model cache is created.
The current normalized image supplies both projections. The initial/final
stage guards, two chronological K16 products per K32 step, F32 accumulator
order, pitch40 mapping/pad zero, and original workgroup barriers remain.

Down fragments retain direct cooperative stores into the genuine320-stride
output. The partial injection fragment cannot use a16-column global store
with a4-column owner. Only `tile_n==0`, row320, and `sg%4==2` store that8x16
accumulator to eight independent128-F32 SLM slots indexed by `sg/4`.
True SG16 acquire/release barriers surround scalar publication of four actual
columns and slot reuse across two M fragments. Other60 columns never publish
or fault, including zero-times-NaN values in padding. No divergent workgroup
barrier is introduced. Existing40KiB operands plus4KiB publication scratch
use45,056B SLM, below the retained49,152B original host envelope.

A single independent finite consumer visits all320+4 logical outputs through
three real buffers (down RW, injection RW, original sticky fault). Each actual
nonfinite value becomes zero and ORs the same original fault. Down/injection
remain separate outputs, with explicit complete disjointness checks against
all source/fault/workspace descriptors. The producer itself has five buffers,
no fault or atomic operations. Native hazard planning sees genuine producer
writes and consumer RW accesses for BOTH outputs.

Production preserves the original active prompt Scope, full `STAGES.gr_input`
identity, full GR arena/alias checks, exact weight geometry and original external
injection owner. The raw implementation never invents a prompt/batch scope.
Counts>=128 use only their complete real M128 prefix. The remaining actual
members use unchanged original down and tiny-injection primitives with an
actual shifted input view and exact output offsets. Count148 means128+20,
not padded/fabricated tokens. Whole prefix+suffix owners, workspace, strides,
grids, capabilities and intervals are validated before first prefix enqueue.
The old tiny workspace is the existing admitted working-stage owner and is
used only for a real suffix; full-prefix calls leave it untouched. Counts<128,
no-injection calls and unsupported hardware retain original paths.

Counters describe actual work and never fake old dense/tiny/prefix calls:
`hc_fused_down_products`, `hc_fused_injection_products` count each combined
prefix dispatch; `hc_fused_prefix_tokens` and `hc_fused_suffix_tokens` record
exact members. `hc_fused_full_down_products` increments only for count>=512
and a complete128 multiple, the cases that previously incremented the original
`gr_wide_tail_products`. Their sum preserves that original logical full-call
coverage. A split148 call does not pretend to be such a full call. With the
six-phase diagnostic enabled, `hc_sample_down` includes BOTH combined products;
`hc_sample_injection` then contains only its diagnostic fence. Do not report
that changed scope as a speedup of a separate unchanged phase.

## Required acceptance

Source adds four focused CPU controls: full two-weight/padding map; complete
output/scratch ownership; rejection of undersized/aliasing complete owners;
and actual compiled module contract. Sixteen new compiler-layout validations
are separate from every original shader/layout population: eight configurations
for producer and finite consumer. Check45,056B/0B actual SLM, SG16 producer,
original eight matrix products, four direct down plus two real subgroup SLM
cooperative stores, no producer atomics, both consumer fault branches,
actual register-load/product/store/fence CFG, and spirv-val. Original kernels
and all original compilation controls remain untouched.

Twelve fixed-public GPU cases compare every bit of BOTH output owning buffers
and the entire sticky-fault owner against unchanged actual down and tiny GPU
primitives. Real original model BF16 tensors, independent authentic owners,
poisoned source-owner padding, output prefix/tail sentinels, exact original
sources, and actual old tiny workspace are checked. Cases cover counts128,
148,512,513, nonzero output offsets, NaN/Inf down weight, NaN/negative-Inf
injection weight, and NaN input in prefix/suffix plus Inf input. Sources stay
in zeroizing RAM; no prompt, image, expert, route, token or tensor values are
persisted in receipts. Reports contain static fixture labels and mismatch
counts only. Original457 GPU controls remain (469 total with this addition),
all original CPU/runtime/shader/layout tests, BOTH immutable original18-script
chains/routing and physical recovery remain necessary before acceptance.
No GPU work or builds were performed by the implementation subtask. Root owns
builds, hardware, protected execution and final validation. No API, model,
context, scheduler, privileged control, frontend, installed binary, release,
GPU reset or reboot changes. Required Vashti changes: none.
