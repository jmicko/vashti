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

Protected fixed-public acceptance can use a user systemd unit without sudo.
Pass the frozen executable, vendor and fixed mode to
`python3 apps/kuno/benchmarks/text/protected-lab-exec.py` under
`systemd-run --user --collect --wait --pipe`, with `MemorySwapMax=0`,
`LimitCORE=0`, `MemoryMax=8G` and the desired lab-only `RuntimeMaxSec`.
The wrapper verifies actual cgroup/core limits before exec; it fails if they
are missing. Set `KUNO_TEST_GGUF` and a fresh `KUNO_PUBLIC_RECEIPT` path on the
unit. The8 GiB cap is an acceptance-test limit, not a matched performance
configuration. Exact run settings and frozen identities belong in receipts;
do not use unit wall time or inconsistent MemoryPeak output as benchmark data.

The source-only `resident-cpu-gpu-expert-probe` registers the bounded original
[CPU/GPU expert numeric operator](../../../../docs/kuno-native-text-cpu-gpu-calibration.md).
It requires `KUNO_LAB_TRANSIENT_IO=1`, `KUNO_LAB_OWNED_ZERO_AUDIT=1` and a fresh
`KUNO_PUBLIC_RECEIPT` destination; missing protections reject before device or
model access. Run the frozen executable through the same protected wrapper,
with the existing original `KUNO_TEST_GGUF`, selected vendor and this fixed mode.
There are no extra case, split, worker or prompt arguments. One already-open
GGUF, one bound graph, the configured client/limits and one shared retirement
owner serve four sequential Numeric-only cases: Q5 down with M10/K0 and M10/K5,
Q8 down with M10/K5 and M10/K10, where M is misses and K is CPU misses. Every
case retains all ten ranks, uses public group0, explicit AVX2/16 distinct
physical workers and Concurrent scheduling. Endpoints and the Q5 mixed case
use the signed/zero-rich public vector, including zero-probability ranks; the
Q8 mixed case uses the other fixed public vector. Unsupported CPU topology/ISA
rejects without fallback. Each producer completes owning maps, numerical gates,
18-page/884,736-byte MAIN erasure, checked I/O cleanup and file consistency.
The returned timing permit is discarded. Only aggregate errors, declared
settings, charges and counts-only cleanup evidence are written after all four
cases complete; an existing receipt is never overwritten. This narrow subset
has source registration only: both-card physical acceptance, other splits,
timing, automatic placement and whole-model CPU dispatch remain unaccepted.

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
resident I/O and remains disabled by default; the short B70 chat ABBA is slower.
For correctness/cleanup audits,
`KUNO_LAB_OWNED_ZERO_AUDIT=1` retains and reads every full main/source page after
erasure, including padding, then checks its readback staging before release.
Both cards inspect all 1,286 main pages/785,186,816 bytes through normal/fresh
owners and queued cancellation. Counts are available only after both arenas
close successfully; failed audit ownership is quarantined until process exit.
Receipts are written after explicit cleanup succeeds. Opaque driver memory and
general private worker/host lifecycle remain outside this proof.

`matmul-probe` configures owned I/O itself and exercises all seven original packed
formats at counts 1/3/5/8 and tiles 1/4/8 (84 numeric cases), with the same nonzero
packed/input/output offsets, irregular strides, inactive tails and full-page
checks. Do not set `KUNO_LAB_TRANSIENT_IO` for this mode. Tile 1 is a source-only
candidate pending independent numeric acceptance on both cards; previous
numeric/full-span receipts cover tiles 4/8. `KUNO_LAB_MATMUL_TIMING=1` retains its
fixed cached eight-vector ABBA comparison against eight legacy cooperative
launches, which showed a cached operator gain on both cards. The isolated
operator does not establish whole-engine speedup; the connected model experiment
is described below and its tile policy is documented in
[the prefill boundary](../../../../docs/kuno-native-text-prefill.md#public-static-projection-tile-experiment).

`moe-word-probe`/`moe-word-bench` test the optional exact-schedule 16-thread down
kernel. `KUNO_LAB_WORD_DOWN=1` can select it in full fixtures, but measured cached
all-ten MoE is slower on both cards/formats. It remains disabled by default.

`KUNO_LAB_ORDERED_PUBLIC_UPLOAD=1` uses moved, owned staging for original public
expert slices, retaining complete source pages until the next completed router
or explicit teardown. It requires owned resident I/O and rejects expert batching.
Every router forces cleanup even under token-level cadence; sources are capped
at 40 MiB and runtime I/O at 64 MiB. Explicit close drains/erases pending source
staging before the optional full-main zero audit, without granting MAIN credit.
Prior staging then has checked erasure/completion proof rather than a zero
inspection receipt. Both cards pass chat, cancellation and full-main audits,
but the paired bang fixture is slower on both; this option remains disabled.

`gr-batch-probe` configures owned I/O itself and checks original attention,
feed-forward and final GR matrices at counts1/3/5/8 and tiles4/8. Leave
`KUNO_LAB_TRANSIENT_IO` unset for this mode. Both cards pass 24 cases against
the unchanged scalar GPU and independent Rust CPU oracles, including irregular
spans, inactive tails and full owning-page erasure. This is an isolated operator
probe without timing or complete causal prefill acceptance. The internal
[prefill plan](../../../../docs/kuno-native-text-prefill.md) records the remaining
transaction and attention/recurrent boundaries.

`gdn-chunk-probe` configures owned I/O before any launch and needs no GGUF. Leave
other resident experiment flags unset. It checks counts1/3/5/8 for odd and actual
Qwen recurrent shapes against the existing GPU and independent Rust core,
including immutable old state, cancellation/invalid-control rejection and full
owning-page cleanup. Both cards pass; actual-shape recurrence is bit-identical
to the scalar GPU in these fixtures. This mode does not batch convolution,
projections, closing gates or the complete model, and emits no timing claim.

`bank96-probe` borrows the original GGUF and owns a separate bounded one-layer
96+10 comparison; leave expert overrides unset for this mode. Both cards pass
Q5/Q8 cross-segment numerical, cancellation, poison and full cleanup checks.
Full resident96 experiments additionally require explicit
`KUNO_LAB_STATIC_CACHE_BYTES` and block MoE; batch/word/ordered/nonresident paths
are rejected. The 18 GiB bank plan fits the tested B70 with explicit6 GiB static
cache and is rejected under the 3060 full-model limit. The short B70 chat passes,
but a later long fixture fails runtime cleanup. No 96-slot default or longer
acceptance is selected; ordinary bank/cache defaults remain unchanged.

`KUNO_LAB_NATIVE_MAX8=1` connects actual layer-major prompt chunks to the original
model owners. It requires `KUNO_LAB_TRANSIENT_IO=1` and accepts an explicit
`KUNO_LAB_NATIVE_BATCH_COUNT=1..8` (default8). Intermediate chunks omit the final
mixer/head; scalar output generation continues from the accepted original state.
Both GPUs pass short bang/chat fixtures, including chat counts1/3/5/8, native GDN/QSA
cancellation, poisoned retry rejection and fresh-owner recovery. Four complete
audits per run inspect1,376 private pages/818,249,728 B. Longer whole-model
contexts, complete options/quality, production private workers and matched
faster-than-Ollama performance remain pending. This flag stays off by default.

The source-only `KUNO_LAB_NATIVE_DECODE=1` experiment continues generation through
native count1 `LastLogits` chunks on that same owner, retaining GPU-f32 PLE. It
requires native MAX8, owned transient I/O, complete zero audits and a compiled
resident bang/chat/rainwater fixture; invalid settings reject before device
creation. Decode, cancellation and poisoned retry use one selected path without
scalar fallback. Receipts record observed successful generation-only native and
scalar steps; all four MAIN audits and exact scalar source-page counts remain
required. Add `--native-decode` to the public runner below to select it. Existing
runners explicitly pin it off. Device/numeric acceptance and a speed benefit are
unestablished; mandatory native maps/proofs and count1 MAX8 overhead remain.
Live runners require all new decode receipt fields. Offline archived-receipt
inspection alone may pass `allow_legacy_decode=True` when all three fields are
absent and scalar decode is selected; archived receipt files are unchanged.

The explicit `KUNO_LAB_NATIVE_DECODE_TILE=1` experiment additionally requires
protected native decode; its default is8 and only1/8 are accepted publicly.
Prefill retains tile8. After completed prefill, the owner selects the public
projection policy once before native count1 continuation; private token counts
do not select shader keys. The public runner accepts
`--native-decode --native-decode-tile 1`, while other modes explicitly pin8.
Receipts retain observed prefill tile8 and record the observed decode tile;
validators require the independently selected tile. Offline legacy inspection
may explicitly accept a missing decode tile only as8, including older counted
native-decode receipts. It never accepts a missing tile1 proof. Device numeric
and speed acceptance remain pending. Existing fresh replay uses a new tile8
owner, and the pre-enqueue cancellation check does not prove a partially queued
tile1 decode failure.

The fixed-public runner selects the **actual `kuno-text-lab` executable from Cargo
compiler-artifact JSON** and freezes its bytes by SHA256. Do not select the old
`kuno-rust-gpu-probe` filename: it can remain in a reused target directory and its
successful exit does not establish any model execution. Use a fresh receipt and
unit name for each run:

```bash
cargo build --locked --offline --release --message-format=json \
  --manifest-path apps/kuno/benchmarks/native-text/Cargo.toml \
  > /tmp/kuno-native-cargo-artifacts.jsonl
python3 apps/kuno/benchmarks/text/run-native-max8-lab.py \
  --cargo-artifacts /tmp/kuno-native-cargo-artifacts.jsonl \
  --model /path/to/existing/ollama/blob --card intel \
  --fixture resident-chat --count 8 --run-id native-chat-intel-unique \
  --receipt /tmp/kuno-native-chat-intel-unique.json
```

The runner checks the protected unit's actual swap/core/host bounds, exact public
fixture partition/output and full audit/cancel/cleanup completion before writing
its run record. Only compiled public bang/chat/rainwater fixtures are accepted;
rainwater compares the full75/128 fixture against the committed original native
public IDs, with fresh full-prompt replay and cancellation/audits. Never use
it to retain private request data. Lab units have a900-second containment bound,
which is unrelated to production inference. Diagnostic durations are not paired
performance samples. See [current model acceptance](../../../../docs/kuno-text-performance.md#connected-native-model-laboratory-2026-10-05).
