# Qwen 2.1 dense MLP experiment

This internal experiment is disabled by default. Exact environment value
`KUNO_EXPERIMENTAL_QWEN_MLP_F16=1` enables it only when the Kuno shim chooses
Qwen 2.1 on Vulkan. Other values, CUDA, and Qwen 2512 retain their prior policy.
It is not an API setting or an automatic performance decision.

The shim supplies an explicit F16 tensor rule for canonical diffusion
`transformer_blocks.<integer>.img_mlp.gate_up.weight` and `img_mlp.out.weight`.
Global `wtype=F16` still leaves tensorwise INT8 weights unchanged. Conditioning,
embeddings, attention, and other diffusion weights remain compressed.

ModelLoader preserves source I8 type, rotation metadata, file identity and byte
counts; only the destination expected type changes. Before worker threads, it
loads small scalar/per-output floating scale arrays. Each converted row folds
its scale, applies the inverse symmetric normalized H4 tensor powers, validates
finite F16 range, then rounds to F16. No complete FP32 weight copy is created.
Linear omits scale parameters and activation quantization for these weights and
requests FP32 GEMM accumulation. This implements the reconstructed dense weight
math; it is not byte-equivalent to the old path, which quantizes activations.

Conversion rejects unsupported shapes/groups, missing, wrong-length,
nonfloating, nonfinite or negative scales, F16 overflow, incompatible/noncontiguous
destinations, and separate logical scale sources. Zero scales reconstruct zero
rows. Scales in other physical shards are supported when one safetensors index
owns both shards; their shared logical FileId retains source revision validation.
Source identity remains the engine's existing size/mtime scheme, not a new hash
check. Model files must remain immutable during loading/execution.

## Memory bounds for the pinned default dimensions

Per layer, `gate_up` has 24576x4096 values and `out` has 4096x12288 values:
144 MiB I8 becomes 288 MiB F16. Across 32 layers, selected resident weights grow
from 4.5 GiB to 9 GiB: **4.5 GiB extra**, in CPU parameter memory and in any
resident compute copies. This excludes unconverted parameters and workspaces.

Converted loading caps workers at four. A largest source buffer is 96 MiB.
Host destinations already own their 192 MiB F16 storage; conversion adds one
FP32 row (at most 48 KiB). Direct nonhost destinations additionally need a
192 MiB conversion buffer per worker: at most 1.125 GiB raw+conversion buffers
across four workers, excluding resident parameters. All per-output scale arrays
for these 64 weights occupy 3.5 MiB in F32; scalar scales need less.

The experiment requires at least **32 GiB available host RAM**, and **14 GiB
GPU budget after the existing free-memory safety margin**. Prefix cache capacity
uses that 14 GiB reserve instead of 9 GiB; it cannot escape into host-cache mode
on Vulkan. These conservative admission bounds are not measured high-water
marks. Mapped raw pages may coexist with owned F16 parameters, so the nominal
4.5 GiB model delta is not a precise physical-RAM peak. Do not enable by default
before measuring owned host/GPU peaks.

## Validation and integration

`mlp-f16-loader-test.cpp` uses only tiny synthetic temporary safetensors and the
real ModelLoader and Linear. It compares reconstructed half values to an
independent dense H4 oracle (either sign of zero is accepted), validates dense
CPU output and FP32 accumulation, proves activation quantization/scales are
absent, and covers mmap, groups 4/16/256, F32/F16/BF16 scales, scalar/per-output
scales including zero, opt-in/scope, invalid metadata, overflow, destination
bounds/contiguity/type, scoped loading, indexed shards, refresh and separate
logical scale rejection. Temporary files are removed on successful completion.
No real model, prompts, images, GPU execution or service changes are involved.

Compile this test like the existing embedding test, additionally including the
native engine's `thirdparty` directory for `json.hpp`. Compile assertions enabled;
link the patched engine and GGML static archives. Running without arguments is
CPU-only. The revised 001 patch introduces `src/model_io/int8_dense.h`; native
build validation must include it with `git add --intent-to-add` before comparing
the exact patch. Vendor patch 002 is unchanged.

Rebuild every engine object consuming `ggml_block.hpp`, especially
`model_builders.cpp`; replacing only loader objects cannot validate runtime
Linear behavior because old inline definitions may remain in the archive.
Then perform bounded root-controlled matrix/device tests, a four-step workflow,
and full image acceptance with memory and timings before deciding deployment.
CPU tests do not establish GPU speed, quality, peak memory or cancellation.
