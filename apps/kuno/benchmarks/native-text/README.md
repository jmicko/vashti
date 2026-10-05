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
