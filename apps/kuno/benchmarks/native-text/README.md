# Native Rust GPU laboratory

This standalone Cargo package is an operator experiment, outside the production
workspace/release dependencies. It reads the existing GGUF through `kuno-text`
and uses Rust-authored CubeCL 0.10 kernels through Vulkan. No C++ model graph,
CUDA/SYCL kernel sources, model downloads or converted weight files are used.

Select exactly one physical discrete adapter by Intel/NVIDIA vendor, rejecting
ambiguous selection. The packed probe checks all seven actual weight formats
against independent CPU decoding/dots with bounded model slices and public
synthetic inputs. `intel bang` or `nvidia bang` runs the complete native graph;
`chat` selects a fixed public chat and `rainwater` the committed baseline.
Arbitrary/private prompts are deliberately not accepted. Results printed by
this program are public synthetic test output, not a production logger.

```bash
KUNO_TEST_GGUF=/path/to/existing/ollama/blob \
  cargo run --locked --release --manifest-path apps/kuno/benchmarks/native-text/Cargo.toml -- intel probe
```

Actual model runs should be contained by the operator in a cgroup with swap/core
dumps disabled, bounded host memory, no other inference and verified free GPU
capacity. The prototype shards matrix uploads into 64 MiB tiles and bounds its
GPU weight cache/reserved allocator bytes. Expert cache indices/access ordering
are transient to the one public experiment. Host intermediate/driver buffers,
GPU erasure on every failure and a persistent private worker still need privacy
and lifecycle acceptance before linking this executor into Kuno.

The `bang` and `chat` fixtures use seed 424242, temperature zero and greedy
sampling (the RNG is unused). Simple greedy equality is only an initial graph
acceptance check; longer contexts, QSA sparsity, options, templates/tool calling,
vision and simultaneous requests remain unaccepted. The first scalar GPU
executor is a correctness prototype; its presence is not a speedup claim.

Additional fixed modes: `matrix-bench` and `moe-bench` alternate cached variants
in ABBA order within one process; `resident-probe` checks residual operators and
complete owning-page erasure; `resident-gdn-probe` checks seven recurrent steps
with GPU-resident old/next state. `resident-bang`, `resident-chat` and
`resident-rainwater` connect GR, complete GDN and all-ten MoE through a
whole-token resident owner, including exact QSA candidate/cache ownership.
Projected PLE retains the Rust host oracle. Both cards reproduce the four-token
reference; B70 also matches the eight-token chat. `resident-qsa-probe` checks
attention through the first actual sparse-selection boundary at context2052.
Both cards pass all six contexts after replacing shader trigonometry with
1MiB of public Rust-generated rotary constants; oracle tolerances are unchanged. This remains
an isolated public-fixture laboratory, outside the installed private worker.

`KUNO_PUBLIC_RECEIPT` optionally writes a new numeric/public-ID receipt, refusing
an existing file. This executable accepts only its compiled public fixtures;
the option is not an arbitrary request logger. `KUNO_LAB_CANCEL_CHECK=1` on a
resident fixture additionally cancels a sequence, verifies whole-arena/host PLE
teardown and poisoned retry rejection, then checks a fresh sequence's first
output using the same public weights. It also cancels after queued GDN and QSA
work at layers0/3 and requires complete arena erasure. Under the earlier public
arena policy, ordinary runtime staging was outside that arena proof; the opted-in
owned-I/O follow-up below covers the specified staging spans. Experiment switches are
`KUNO_LAB_SMALL_CPU=1`, `KUNO_LAB_BLOCK_MOE=1`, `KUNO_LAB_CPU_GDN=1`, and
`KUNO_LAB_BLOCK_KERNEL=0|1` (generic/all shapes). Without the block-kernel
override, only large outputs use block decoding. The tested full rainwater run
explicitly selected small CPU projections, large-output block decoding and the
block MoE kernel. See the performance guide for timings and unaccepted work.

The earlier public arena policy is not suitable for real requests. The lab now
vendors three pinned CubeCL crates and provides an explicit owned-I/O policy;
see [the residency audit](../../../../docs/kuno-native-text-residency.md) and
[runtime ownership](runtime-ownership/README.md). This remains a compiled public
fixture runner, with no production private worker or request API. Never
specialize private tokens, positions or routing as compile-time shader constants.

`KUNO_LAB_TRANSIENT_IO=1` on a full resident fixture configures owned runtime
uploads, uniforms and readbacks before the first launch. Private source pages
remain owned through their consumers; map leases drop before checked cleanup.
`KUNO_LAB_PREFILL_LAST_HEAD=1` omits unused intermediate vocabulary projections
while keeping complete layer/state/finite acceptance. Both cards pass multi-token
chat; B70 preserves the full 128-token prior native fixture. This longer native
output is not bit-identical to Ollama.

`KUNO_LAB_IO_TOKEN_CHECKPOINT=1` additionally defers intermediate I/O erasure to
the accepted final token fence under the unchanged 64-MiB runtime cap. Main-pool
fence credit cannot reclaim tracked runtime spans. This cadence requires owned
resident I/O and remains an optional experiment. For correctness/cleanup audits,
`KUNO_LAB_OWNED_ZERO_AUDIT=1` retains and reads every full main/source page after
erasure, including padding, then checks its readback staging before release.
Both cards inspect all 1,286 main pages/785,186,816 bytes through normal/fresh
owners and queued cancellation. Counts are available only after both arenas
close successfully; failed audit ownership is quarantined until process exit.
Receipts are written after explicit cleanup succeeds. Opaque driver memory and
general private worker/host lifecycle remain outside this proof.

`matmul-probe` configures owned I/O itself and validates all seven original packed
formats at counts 1/3/5/8 and tiles 4/8. Do not set `KUNO_LAB_TRANSIENT_IO` for this
mode. `KUNO_LAB_MATMUL_TIMING=1` adds a fixed cached eight-vector ABBA comparison
against eight legacy cooperative launches. Both cards pass numeric/full-span
checks and show a cached operator gain. True layer-major causal prefill and a
whole-engine speedup remain unimplemented; sequential step loops are not batching.

`moe-word-probe`/`moe-word-bench` test the optional exact-schedule 16-thread down
kernel. `KUNO_LAB_WORD_DOWN=1` can select it in full fixtures, but measured cached
all-ten MoE is slower on both cards/formats. It remains disabled by default.
