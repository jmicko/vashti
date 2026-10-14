# Complete Strata Intel inference rewrite: completion gate

Approved 2026-10-14. This is backend engine work, not gateway/API/UI/release work.
The user requires a complete coherent Rust port before live speed testing. This
checklist is a source-to-source audit, not a sequence of throughput experiments.
Each row needs its connected consumer, real ownership/placement and state contract;
adding an unused helper or optional switch does not complete a row.

## Reference and execution scope

Pinned MIT upstream `fb58e0dbc8399662c0e47c76578c6e878b14f6cf`, checkout
`/home/jmicko/.codex/worktrees/kuno-native-split-gguf/strata-research-20261009`.
Follow its **Intel** single-device inference path: original full 512-expert
Qwen3.8-Flash-Next DASLab IQ3_S, original Q2 MTP sidecar, real router and target
verification. Intel B70 is the selected device. CUDA/HIP-only kernels, other model
families, web server/UI, distributed GPUs, optional content-persisting conversation
caches and benchmark programs are not required to reproduce this inference path.
Do not use those scope boundaries to omit prompt prefill, sampling, state commit,
CPU/GPU residency, or a dependency of the working measured Intel path.

Rust must own model loading, graph execution, request state and inference kernels.
Generated WGSL/SPIRV and Rust/ash Vulkan are the explicit backend substitution for
SYCL kernels/queues. No foreign inference graph/library is linked. Preserve MIT and
GGML notices. Substitution must preserve buffer placement, data movement, grouping,
work sharing, ordering and state transitions; an alternative schedule must be
recorded as a remaining gap until reviewed, not silently described as parity.

## Connected dependency checklist

Current state is **incomplete**. Entries below are preliminary source findings;
all need a final audit of actual connected dispatches, not just file names.

| Component | Pinned upstream | Current Rust connection and required completion |
| --- | --- | --- |
| Artifact geometry, packing and tokenizer | `sycl/src/core/weights.cpp`, `src/io`, native expert layout, program model setup | Rust GGUF/tokenizer/binding exist. Audit format-specific load/conversion, chat-template and sampling behavior; original files remain shared/read-only. |
| Exact dense placement and expert budget | `sycl/src/program/generate.cpp` model/cache setup; `sycl/src/core/expert_cache.cpp::open_sized` | Exact dense roots exist. Uniform 240 slots/layer differs from global ranked variable-size allocation; implement the connected global plan and consumers. |
| Low-RAM host complement and promotion | `generate.cpp` source/adaptive tier; `sycl/src/core/verify.cpp` fetch/rebase | Original host complement, direct exchange and GPU selected fetch exist. Audit full byte ownership and asynchronous safe publication against upstream; do not duplicate the whole model in RAM. |
| GPU route/group planning and CPU contribution | `verify.cpp` router, `resident_plan`, doorbell/pool/host plans; `src/kernels/cpu` | Real GPU router exists. Audit the selected working Intel all-GPU-fetch versus host-pool mode and reproduce its actual mode, including overlap/merge where active. |
| Native expert math and grouped rows | `sycl/src/kernels/cuda/iq_kernels.dp.cpp::row_entries`, `native_gu_kernel`, `native_down_kernel` | Packed integer math/BDA roots exist. Separate GU roles, 4/2/1 chunks, activation, intermediate storage and downstream consumers are now selected together in the Intel core; all 158 WGSL/SPIR-V variants validate, including 105 physical-pointer checks; numerical/model execution acceptance remains pending. The Vulkan port uses separate fixed chunk launches with GPU guards, not upstream row_entries within one launch; audit that substitution explicitly. |
| GR write/read/norm/down/up/injection | `sycl/src/kernels/cuda/fused_gr.dp.cpp::fused_gr_read_multi`; `verify.cpp::gr_read_group` | Existing Rust GR still uses separate projections/pointwise stages. Port pending-write norm, sliced down/reduce and up/mix/injection with genuine resident BF16 weights and charged partial scratch together. |
| GDN projection/conv/gates/recurrence | `verify.cpp`, `cuda/native_gdn*.dp.cpp`, `cuda/fused_gdn.dp.cpp` | GDN retained-state replay exists. Audit actual convolution/L2/alpha-beta/recurrence/output-norm fusion, grouped projections and retained commit inputs against pinned Intel source. |
| QSA projection/norm/RoPE/KV/indexer | `verify.cpp`, `cuda/native_qsa*.dp.cpp`, `cuda/qsa*.dp.cpp` | Causal Rust QSA exists. Audit fused norm/RoPE, KV format/append and block selection/gather, grouped consumers and tail repair; explicit F32 versus quantized KV difference needs resolution. |
| PLE source rows, projection and history | `verify.cpp`, `ngram`/`ple` kernels, PLE table reader | Original grouped reads/hash/history exist. Audit asynchronous deduplicated source reads, grouped BF16 projections and ordered history snapshots/commit. Never persist request n-grams. |
| Prompt prefill and streamed experts | `sycl/src/prefill/prefill.cpp`, `gemm.dp.cpp`, MoE MMQ/Intel implementations | Rust bounded graph prefill is not full prompt-path parity. Port connected chunk budget, streamed layer staging and dense/expert batches, including prompt history consumed by MTP. |
| MTP loading/prefill/draft/head | `src/core/mtp.cpp` as built by Intel target, Intel kernels | Original Q2 sidecar and public draft vocab exist. Audit native layout/conversion, draft graph/attention and prompt-state initialization; arbitrary text and full supported draft head must work. |
| Suffix proposal and draft policy | Intel speculation implementation and `DraftPolicy` | Request-local Rust suffix/cost controller exists; audit supported window lengths and coupled sampling rather than fixing to four public-fixture rows. |
| Target verification and accepted-prefix commit | `sycl/src/core/verify.cpp` checkpoint/commit methods | Real greedy verification/state-only repair exists. Audit all active windows and acceptance cases against GDN/QSA/PLE/positions; independent correctness oracle remains missing. |
| Graph capture, queues and synchronization | `verify.cpp`, session/mtp graph construction, Intel queue/event implementation | Native Vulkan owned recording/replay exists. Audit stable complete recordings, transfer dependencies, scratch lifetimes and actual overlap; a sequence of independent Wgpu launches is not equivalent. |
| Request lifecycle and arbitrary input | upstream CLI/session/inference entry points | Current Rust public-fixture gate is not a general engine. Connect bounded arbitrary prompts and request-local sampling/history, with no persistent content, protected memory and orderly erase/close. No gateway/UI work required. |

## Validation and the speed gate

1. Keep the connected port selected as one engine configuration. Fix compile,
   shader, numerical and ownership bugs as they arise. Do not retain a partial
   upstream behavior as default-off merely because an unfinished run was slower.
2. Interim GPU execution is allowed only to prove correctness/completion, with
   swap/core dumps disabled and actual resource recovery checked. Disable operator
   profiling. Do not compute/present a throughput ranking or run comparative speed
   trials during this stage. Fixed public fixtures may be used; private content
   never enters receipts/logs.
3. Final completeness audit must identify each actual call path, source revision,
   resident owners, dependent consumers and every resolved substitution. Remaining
   upstream inference dependencies mean the rewrite is incomplete.
4. Then run end-to-end correctness/completion tests: prompt prefill, count1 and
   supported multirow verification, acceptance/rejection/repair, EOS and capped
   generation, and recovery. Correctness failure blocks performance acceptance.
5. Only after (3) and (4) pass, run live throughput comparisons on the assembled
   engine and fix performance bottlenecks. 30 accepted tokens/s remains the desired
   result; a speed number does not itself prove completeness or correctness.

No frontend changes, release or installation is authorized by this checklist.
Historical artifacts/receipts remain evidence, not the next direction. If root
credentials expire, document where the connected implementation stands and stop
at a safe boundary instead of spending time on privilege workarounds.

## Current implementation block

`strata_expert_roles.rs` connects gate/up/DOWN 4/2/1 chunks to the same authentic
physical source roots and one HOST fetch sequence. Gate/up borrow two 640-value
planes from each existing 2560-value PART row; the activation kernel consumes and
zeros both planes before original DOWN preparation/consumption. Per-role stamps
use the existing status owner and are restored for DOWN. All intermediate owners
remain in PrivateArena; no extra private weight/input roots are allocated. Eight
extra INFO owners/layer are charged together. Core selection is unconditional;
there is no speed-driven fallback. Other incomplete audit rows remain required.

The protected runner now requires `--execution-check` and rejects operator
profiling while the rewrite is incomplete. Its source receipt labels the run as
execution/correctness only, never throughput acceptance. Release compilation, strict core Clippy, fmt/whitespace, runner syntax, the
158-variant shader set and original direct-device regression (1,346 passed,
1 ignored) pass. Runner refusal of speed/profile invocations was exercised
without root, service calls, output creation or GPU access. No GPU inference run was performed in this
implementation block. Full model numerical execution remains pending.
