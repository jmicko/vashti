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
Long-position NVIDIA arithmetic is still under correction at this checkpoint. This remains
an isolated public-fixture laboratory, outside the installed private worker.

`KUNO_PUBLIC_RECEIPT` optionally writes a new numeric/public-ID receipt, refusing
an existing file. This executable accepts only its compiled public fixtures;
the option is not an arbitrary request logger. `KUNO_LAB_CANCEL_CHECK=1` on a
resident fixture additionally cancels a sequence, verifies whole-arena/host PLE
teardown and poisoned retry rejection, then checks a fresh sequence's first
output using the same public weights. It also cancels after queued GDN and QSA
work at layers0/3 and requires complete arena erasure. The ordinary runtime's private staging
copies remain outside that arena proof. Experiment switches are
`KUNO_LAB_SMALL_CPU=1`, `KUNO_LAB_BLOCK_MOE=1`, `KUNO_LAB_CPU_GDN=1`, and
`KUNO_LAB_BLOCK_KERNEL=0|1` (generic/all shapes). Without the block-kernel
override, only large outputs use block decoding. The tested full rainwater run
explicitly selected small CPU projections, large-output block decoding and the
block MoE kernel. See the performance guide for timings and unaccepted work.

The private arena proves erasure only for its own full allocation spans.
CubeCL/wgpu also own upload, uniform and readback copies; their sanitization
remains unaccepted. See [the residency audit](../../../../docs/kuno-native-text-residency.md)
before exposing a persistent worker to real requests. Never specialize private
tokens, positions or routing data as compile-time shader constants.
