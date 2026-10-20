# Prompt GDN scalar-state candidate (2026-10-20)

This Kuno-only candidate expands the complete selected CB32 recurrence path.
It has CPU compiler acceptance, but no GPU acceptance, speed result or
installation. APIs, model identities, context limits, privacy and Vashti are
unchanged. Full request state still belongs to the original wiping arena.

## Actual selected reference

MIT Strata `fb58e0d`, `sycl/src/prefill/kernels.dp.cpp:2634`, selects the
software-pipelined CB32 column recurrence by default. The optional key-head
kernel is explicitly opt-in on Xe at1228–1232; the retained fresh reference
does not set that variable. The reference uses128 threads per block, four
32-column blocks per value head, four row groups, and32 state rows per thread.
Its `gdn_rec_cols_pipe_kernel` at880–990 explicitly unrolls state load, KV32,
state-update/readout32 and state writeback. Its token loop remains causal and
serial. The Rust geometry and token pipeline already match this path.

The actual original Rust `RecurrenceFault` generated shader, lowered through
the existing pinned Naga dependency, retains a function-storage `f32[32]`
state array and all five loops. That is a concrete source/compiler difference;
whether the native GPU compiler spills or later unrolls it remains unproven.
No speed conclusion can be drawn from this difference alone.

## Coherent expansion

`strata_prompt_gdn_specialize.rs` expands only generated `Recurrence` and
`RecurrenceFault` source. The32 states become separate scalar variables and
all four fixed-extent loops are expanded together. Initial loads, each ordered
KV FMA, each state-update FMA followed by the readout FMA and original finite
check/atomic, and final writeback stay in ascending row order0..31.

The existing causal time loop, next-input prefetch, shared Q/K writes, six
static barrier sites (five per token plus initialization), row-group partial
sums, original state layout and old/next ownership, finite checks and sticky
fault/status closures remain unchanged. There is no guard aggregation,
additional32-register K cache, reassociation, approximation, fast math, new
subgroup assumption or binding/launch/capture change. Base WGSL is unchanged.
Other prompt stages, ordinary decode and separate MTP recurrence source remain
unchanged. The original generated scalar source is retained internally for a
future compiled-public golden control.

For a4096-token call with48 value heads, the recurrence executes about9.66
billion FMAs and3.22 billion state finite checks per layer. Removing the
dynamic loop/index machinery could matter even when GPU memory traffic is
unchanged. Those counts are operation counts, not a timing model; the original
guards are retained despite their cost.

## CPU evidence and required next gate

```sh
python3 apps/kuno/benchmarks/text/test-prompt-gdn-specialize.py \
  --deps-dir /media/jmicko/faststore/kuno-development/root-native-text-target-20261005/release/deps
```

All three tests pass: source/FMA/fault ordering, unchanged other prompt stages,
and actual Naga validation/SPIR-V lowering of both selected variants. The test
uses only existing dependency artifacts, with no Cargo/shared-target build,
model, GPU, root command or service operation. Both emitted modules also pass
standalone `spirv-val --target-env vulkan1.3`.

Actual fault-kernel `OpLoopMerge` count falls5→1, the function state array
disappears, and all six static `OpControlBarrier` instructions remain. SPIR-V
grows2833→11524 words. Native register/instruction pressure may therefore
offset some benefit; GPU evidence remains required.

Next gates: the original numerical controls, a real-arena fixed-public
candidate-versus-original scalar recurrence control covering complete output
and next-state owners/faults/tails, full request erasure/recovery, then matched
public prompt measurements. Do not weaken original IEEE/denormal, numerical,
fault, ownership or privacy constraints to select the candidate. No new decode
throughput improvement is claimed.
