# Experimental SYCL materialized-F32 residual fusion

Source candidate based on combined production tree `f6aa880`, engine
`88411ef1e0688ff2df1010aeeb5d92b2d8cea2be`, and GGML
`4bf5f6000653b7881d00963cd6ddb665ccd62a8d`. Patch 001 remains
`ed75739ccb193099f23d4600cc4a8846f9d17fe6c2cec7591a7c9cb30c2d0134` and
patch 002 remains `7298590faed44cad4733c78f6f6c2bf261b647712b5ec5f75840da450f0067dc`.
Patch 003 adds this experiment while preserving the guarded SDPA candidate.
The accepted packaging/installation pins are not advanced by this source work.

## Scope and arithmetic

Qwen21 performs two residuals per layer as `ADD(residual, MUL(activation, gate))`
at `src/model/diffusion/qwen_image_2_1.hpp:290` and `:309`. Warm prefix REUSE
has no prefix rows, so these are adjacent operations. Prefix STORE/NONE paths
with a concatenation between the operations retain their existing execution.

The candidate adds a SYCL backend fusion only for F32 contiguous two-dimensional
activation, residual and output arrays, with an F32 contiguous per-channel row
coefficient and `ADD(residual, MUL(...))` operand order. Both actual Qwen21
residual calls use that order. Reversed ADD operands retain the legacy path
because the exact device control comparisons exposed NaN payload/sign differences.
Full arrays must not be views. The graph's existing fusion check
requires one intermediate consumer, no intermediate output, matching shapes and
both compute flags. An additional scan of graph nodes and leaves rejects direct
or nested views of the intermediate; the existing use count excludes view-only
escapes. Runtime checks reject null/misaligned/wrapping
ranges, partial output/input aliases and every output/coefficient overlap.
Exact full-array output aliases with activation or residual are safe. Read-only
input overlaps are allowed. Device and all accessed buffer types must match the
current Intel Level Zero GPU. Any rejected case retains the old two kernels.

The next prepared kernel uses the original plain `op_mul`/`op_add` helpers in the
original ADD operand order. A volatile F32 product preserves the former storage
boundary, and a kernel-local pragma disables contraction. Device equality still
requires the complete exact oracle; it is not inferred from these expressions.
The prior IMF rounded-operation candidate failed special-value comparisons and
remains preserved as an unselected historical artifact. No normalization
reduction, quantization, GEMM accumulation,
precision, model/workflow setting, API or cache lifetime changes.

There is no new buffer allocation, host copy, retained state or synchronization.
Each work item reads the two full-array values and its channel's gate before
writing one output. The two-dimensional launch avoids per-element integer
division for the channel index. ADD is skipped only after the fused helper
successfully submits work; shape/device/buffer/range rejection submits nothing.

For a target-only `8214x4096` hidden tensor, one F32 array is 134,578,176 bytes.
Two residuals remove four full-array transfers per layer: about 802.15 GiB over
32 layers and 50 diffusion calls, plus 3,200 launches. The analogous 4056-row
operation count is about 396.09 GiB over 50 calls (25 CFG4 steps with separate
positive/negative forwards). Actual prefix STORE, segmentation and allocator
behavior determine eligibility. These are avoided-operation counts, not a
measured speed improvement or proof of the complete workflow's dispatch.

## CPU checkpoint and prepared device verification

`sycl-mul-add-policy-test.cpp` initializes no backend. It covers both ADD orders,
odd/public dimensions, wrong ops/types/strides/shapes/batches, null sources,
zero/negative/overflow dimensions, full-array views, permitted coefficient views,
null/misaligned/wrapping pointers, exact and partial aliases, boundary-adjacent
ranges, coefficient overlap, read-only overlap, output/multiple-use/view escape
and missing compute flags.

`sycl-mul-add-arithmetic-test.cpp` defaults to CPU only. Its independent reference
rounds at both former F32 storage boundaries and checks the separately executed
graph. Synthetic controls include a triple that distinguishes FMA from two
rounded operations, overflow/cancellation, subnormals, signed zero and distinct
positive/negative quiet NaN payloads. NaN comparison against the scalar CPU
reference checks classification; old-versus-candidate device comparison checks
all bits, including payload and sign. Inputs remain synthetic and in RAM.

GPU mode exists only in a SYCL-linked build with `KUNO_SYCL_FUSION_DIRECT=1`, and
requires explicit `--device SYCL <PCI-address>`. It selects exactly one matching
GPU/backend before initialization and verifies the backend GUID before accessing
its context. Each variant uses a fresh graph and external input buffers. The
legacy graph marks MUL as an output, forcing the two old kernels. The candidate
must pass the actual backend fusion policy and successfully invoke the real
fused helper; it then restores/reuploads inputs and executes the eligible real
graph. This prevents an unchanged fallback alone from satisfying the arithmetic
test. Archive provenance must bind the test to the candidate helper and graph
dispatch object. Both ADD orders, exact output aliases and public 4056/8214-row
shapes are covered. Outputs/raw inputs are compared in RAM, workspace is capped
at 1 GiB and the test requires 64 MiB of live free-memory headroom. Confirmed
numerical errors synchronize and unwind resources; device backend uncertainty
exits before caller unwind, as described in the v5 checkpoint below. No GPU
numerical assertion aborts the process, core dumps/backend logs are disabled and
destructor exceptions are contained.

CPU commands executed after the owning task cleared compilation:

```sh
candidate_root=/home/jmicko/.codex/worktrees/kuno-comfy-speed-review/vashti/target/fused-residual-v1
standard_root=/home/jmicko/.codex/worktrees/kuno-comfy-speed-review/vashti/target/shared-rope-standard-v2
mkdir -p "$candidate_root/compiler-temp"
export TMPDIR="$candidate_root/compiler-temp"
c++ -std=c++17 -O1 -DGGML_MAX_NAME=160 \
  -I "$candidate_root/source/ggml/include" -I "$candidate_root/source/ggml/src" \
  -I "$candidate_root/source/ggml/src/ggml-sycl" \
  crates/kuno-image/native/sycl-mul-add-policy-test.cpp \
  "$standard_root/build/ggml/src/libggml-base.a" -lgomp -lpthread -ldl -o "$candidate_root/policy-test"
c++ -std=c++17 -O1 -ffp-contract=off -DGGML_MAX_NAME=160 \
  -I "$candidate_root/source/ggml/include" -I "$candidate_root/source/ggml/src" \
  -I "$candidate_root/source/ggml/src/ggml-sycl" \
  crates/kuno-image/native/sycl-mul-add-arithmetic-test.cpp \
  -Wl,--start-group "$standard_root/build/ggml/src/libggml-cpu.a" \
  "$standard_root/build/ggml/src/libggml-base.a" -Wl,--end-group \
  -lgomp -lpthread -ldl -o "$candidate_root/arithmetic-cpu-test"
```

Fresh patch check/apply/full-diff equality passed against a separate pinned GGML
clone under `target/fused-residual-v1/proof-ggml-cpu`. Engine diff remains exactly
patch 001. The ignored persistent source receipt is
`target/fused-residual-v1/source-receipt.json`; it records pins, paths, patch and
test source hashes and execution boundaries. Independent source review of the
prepared candidate found no guard/lifetime blocker. The CPU policy run then
exposed the view-only escape described above; the guard and its direct/nested
regressions now pass. The policy binary passed 141 checks; the arithmetic binary
passed all eight CPU cases. Its invalid explicit-device invocation was rejected
before backend initialization with phase 1. Both tests disable core dumps.
The CPU run verifies topology/pointer policy and the two materialized CPU F32
operations against the independent scalar reference. It does not execute the
new SYCL kernel or establish its device arithmetic, dispatch or performance.

Persistent receipt `target/fused-residual-v1/cpu-test-receipt.json` records exact
successful compiler commands, versions, binary/source/archive hashes, safe
numeric output and fixed initialization rejection. The initial policy link
needed `-lgomp`; the initial arithmetic compile needed the pinned allocator's
fifth size-output argument. Both harness corrections are included here.
At the 2026-10-01 checkpoint, native compilation and all GPU/service execution
remained unperformed. Work stopped when the user reclaimed the GPUs. This
historical CPU receipt remains intact; the subsequent source build is below.

## Persistent native build and safe oracle (2026-10-02)

After explicit compilation clearance, three SYCL units (`binbcast.cpp`,
`fusion.cpp`, `ggml-sycl.cpp`) compiled sequentially with the recorded original
producer flags. They replace exactly three members of the guarded SDPA archive;
the other 97 members remain byte-identical. Source comparison against the f6
overlay changes only those units, the new helper declaration and policy header.
Engine archive `56fc04019678ae50f35407d6b0891bdc1882720ac2041d667e6b1001e3be4e6e`
and the recorded GGML base/CPU/registry archives and cache metadata are reused
unchanged. Fresh pinned patch application/full-diff proof passed again.

The first IMF rounded-operation candidate SYCL archive is
`0f70efbfb79100938a5b4d6c1485adb29b61693a4601f17316c1040e111efc3d`.
Its independently linked private ABI 2 worker is
`cbac1164186ede02d5098e8c31a26cfdd8941e985bff84bdd40a719ec4f1cd6b`.
Their producer receipts, exact commands and member hashes are under ignored
`target/fused-residual-v1/native-v1`. The Rust build used an independent copy of
the earlier cache, rebuilt `kuno-image`/`kuno`, and post-verified the accepted
worker and native inputs. The worker has no RPATH/RUNPATH and was not executed.
It is a private worker, not a standard gateway artifact.

Root source review held the initial linked oracle before execution because its
destructors could free resources after synchronization failed. The test-only
repair shares `backend_safe` across all resources. Thrown backend calls and
unsuccessful submission/compute outcomes mark ownership uncertain. An uncertain
state prevents subsequent calls, waits and dependent frees; resources remain
for process exit. Safe numerical mismatches retain ordered cleanup. Final
explicit backend release prevents reporting success after failed cleanup.
This preserves the pinned backend's `wait()` semantics and does not claim that
successful synchronization detects deferred device errors or handles a backend
that exits internally instead of throwing.

The repaired oracle is under `target/fused-residual-v1/oracle-safe-v2`, SHA-256
`05a73137162be232aba0ede2403cc9faf1876985b637febd5c1e0816444a4f7f`.
Its matching CPU binary passed eight arithmetic cases. The new CPU-only
`sycl-mul-add-failure-test.cpp` passed nine separate-process fixtures: invalid
entry, thrown initialization, safe numeric failure, query/submit throw, false
submit/compute status, synchronization failure and cleanup failure. Fixtures
verify cleanup order for safe failure and zero dependent frees/retries after
uncertainty; no GPU/backend registry is initialized. Build/CPU receipts record
test source hashes and exact commands. The earlier oracle remains preserved,
unexecuted and superseded.

The flat ABI 2 runtime copies all 63 vendor library/notice members unchanged
from the accepted resident closure and replaces only the worker. A candidate
manifest binds the unchanged engine/GGML pins and current patch hashes. Packaging
uses the unchanged helper with a recorded candidate-local pin override; no
production packaging pin is advanced. Runtime/manifest/archive hashes are in
`native-v1/runtime-receipt.json`. This fusion overlay sits on the still-unresolved
f6 shared-RoPE/aligned-KV overlay; comparison against that f6 overlay isolates
fusion effects. Device arithmetic, pixel consistency, dispatch and speed remain
root-owned validation. The fusion is unselected and the installed runtime is
unchanged.

### Exact mismatch categories

Root's protected first device run reported four passing finite cases, followed
by a normal phase-11 failure on the control set containing nonfinite/subnormal/
signed-zero values. No new kernel fault or uncertain cleanup was reported.
The scalar source review cannot identify which category differs: the old
kernels use plain `op_mul`/`op_add`, while the candidate calls the reviewed SDK's
external `__imf_fmul_rn`/`__imf_fadd_rn` entrypoints. Their control-value behavior
needs the actual device comparison; no tolerance or exception was accepted.

Diagnostic oracle `target/fused-residual-v1/oracle-diagnostic-v3` retains the
safe teardown and exact comparison. Its single fixed mismatch line reports
integer counts for normal/zero finite differences, subnormal differences,
signed-zero differences, two NaNs, a NaN-class difference and infinity
differences. These six categories partition the total. NaN payload/sign and
zero/nonzero-class counters are additional subsets. No bits or floating values
are printed, and comparison still throws the original phase and exits with 2.
Eight CPU arithmetic cases and ten separate CPU fixtures pass, including a
synthetic pair covering every category, its partition/subsets, an unchanged
pair and strict rejection. Source/command/binary hashes and all safe numeric
results are in `oracle-diagnostic-v3/build-receipt.json`. Native fusion objects,
worker and runtime bytes are unchanged by this test-only diagnostic.

### Prepared plain-operator alternative

Root's categorical device rerun reported 455 mismatches: finite/subnormal/
infinity differences were zero, signed-zero differences were 65, both-NaN
differences were 325 (260 payload and 195 sign differences), and NaN-class
differences were 65. Thus an exception for NaN payloads alone cannot establish
correctness. The run exited normally with phase 11; no new kernel fault was
reported. Strict comparison is retained.

An independent source tree under `target/fused-residual-plain-v2/source` now
prepares the plain-operator alternative above. Only `binbcast.cpp` changes from
the first candidate source. Original source/build/worker/runtime and numerical
diagnostic artifacts remain intact. The new patch is experimental; neither
compilation nor device behavior is verified at this preparation checkpoint.
Volatile materialization may cost private storage/spills and requires a measured
comparison after correctness passes; it introduces no GGML buffer allocation.

Controls have 21 public synthetic indices. On failed legacy/candidate comparison,
the prepared oracle additionally prints a bounded `index`/`count` line for each
affected control, without input/output values. A CPU fixture checks repeated
indices across two rows and the duplicated controls in columns 21/22. Compilation
is held during the protected profiler window. Full bit equality, input integrity,
both ADD orders, aliases, public shapes and safe failure behavior remain required.

### Plain-operator build checkpoint (2026-10-02)

After the paired SDPA oracle compiler window ended, source commit `4296d4a`
passed 141 CPU policy checks, eight CPU arithmetic cases and ten separate-process
failure fixtures. The categorical fixture also verifies the control-index
counts across repeated controls and two rows. The build driver's first attempt
used two incorrect fixture names; that incomplete receipt is preserved under
`target/fused-residual-plain-v2/native-v1`. Correct fixture names pass in the
independent completed directory `target/fused-residual-plain-v2/native-v2`.

The completed build recompiles only `binbcast.cpp` with the original producer
flags and one compiler job. Exactly `binbcast.cpp.o` changes in the 100-member
SYCL archive; the other 99 members remain byte-identical to the preserved IMF
candidate. Engine archive `56fc04019678ae50f35407d6b0891bdc1882720ac2041d667e6b1001e3be4e6e`
and the GGML base/CPU/registry archives remain unchanged. The pinned source and
fresh proof clone match patches 001/002 and plain-operator patch 003
`ed671f061b8dce0c0f566a2b7455b2f744d1fb2546cafecc0e77fd905d1910f0`;
source and all borrowed inputs were verified again after linking.

The new SYCL archive SHA-256 is
`933b4b843e93a3f418f30e801f815c3382a13075ef69d08bb7f8636f5a1c4b58`.
The independently linked direct/real-graph oracle is
`88be0b0844f64d69314128876eeb8ceb978219e353afc6120e4ca6670b5577c9`
(84,668,032 bytes), with no RPATH/RUNPATH. Exact commands, member hashes,
compiler/version/environment, CPU outcomes and source proof are in
`native-v2/build-receipt.json`, SHA-256
`ac7cb9cf488781bf3e4a0c14c1e1e10a39a3d2443d5788c19ebeb77b1eae5503`.
The compile reports only the existing DPCT helper warnings. This checkpoint
does not execute GPU code, rebuild/package a new private worker, or change the
installed runtime. Exact device equality and any speed benefit remain unverified;
the candidate remains unselected. The owner performs the protected device test
before deciding whether another worker/runtime build is warranted.

### Host sanity diagnosis and strict test build

The owner's first plain-operator device run reached phase 12 after four finite
cases: the first special-value legacy/direct/real-graph comparison passed its
exact bit check before the host FMA sanity check failed. Test-only commit
`695605f` adds bounded host-control booleans before backend initialization and
only on that existing host failure. The diagnostic run reported `zero=1`,
`positive=1`, `fma_distinct=0`, `nearest=1`, `ftz=1`, `daz=1` at both points.
This failure already exists before backend initialization; it does not establish
that device work changes the host floating-point environment.

The original oracle's Intel driver enables unsafe/reciprocal math,
preserve-sign denormal behavior and no-rounding assumptions despite
`-ffp-contract=off`. The test-only relink explicitly adds `-fp-model=strict`.
Actual host compiler commands now have rounding-math and strict exception
behavior, without those unsafe/denormal/no-rounding flags. An Intel-compiled
CPU-only version with the same strict host flags passes eight arithmetic cases
and eleven failure fixtures, including the new independent host-control
fixture. Its pre-init check reports `zero=1`, `positive=1`, `fma_distinct=1`,
`nearest=1`, `ftz=0`, `daz=0`; its dynamic dependencies contain no GPU runtime.

The strict native oracle is under
`target/fused-residual-plain-v2/oracle-strict-host-v4`, SHA-256
`f9651cb344d22ef57fee82b26ea4fd0645b7bc420ea2926a2d64be2ae3a686d3`
(84,668,296 bytes). Its receipt SHA-256 is
`0dd7af7a3c00bb93c13cd4b7f8b30abcc471d140dced7672b23ed0e5ce1faf01`.
Only test translation units compile; device archive `933b4b843e93a3f418f30e801f815c3382a13075ef69d08bb7f8636f5a1c4b58`
and every borrowed native input are verified unchanged. The device frontend
reports that rounding/exception modes are unsupported for that target; this
checkpoint changes no production kernel object. All twenty device cases,
exact comparisons, raw-input integrity and safe teardown remain required.
The strict device oracle was not executed by this builder. This repair changes
host test assumptions and accepts no device mismatch or tolerance.

### Residual-first eligibility and observed routing

The strict device run passed fourteen cases before the first reversed special
control pair failed exact phase 11. All ten residual-first cases passed,
including special values, aliases and public 4056/8214 rows. The reversed failure
had 195 both-NaN differences (all payload differences; 130 sign differences) and
zero finite, subnormal, signed-zero, NaN-class or infinity differences. This does
not authorize accepting those NaN changes. The approved next candidate narrows
the common geometry guard to require MUL at ADD source 1 and residual at source
0. Graph policy and the direct helper's memory guard inherit that requirement;
the reversed order submits no fused work. Kernel expressions remain unchanged.

The prepared oracle retains all twenty cases. Residual-first candidates require
the real direct helper and eligible backend graph. Reversed candidates leave
the MUL unmarked so the general GGML fusion predicate still passes, then require
the narrow/backend predicates to reject it. Their actual backend graph runs both
legacy operations, including the two exact output aliases, and must reproduce
the forced-materialized baseline bit for bit. Unsupported reverse orders are
never forced through the fused helper.

Test-only GNU link wrappers observe three existing externally called host
symbols: the fused helper and legacy MUL/ADD launch functions. Each wrapper
delegates to its original function. Scoped thread-local counters live solely in
the oracle: an eligible direct/real-graph call requires one fused attempt and
submission, with zero legacy MUL/ADD calls; a rejected or baseline graph requires
zero fused attempts/submissions and one call to each legacy operation. A CPU
fixture verifies nested scopes and exception restoration. No production
counter, content logging, cache or persistent state is added. Exact numerical
comparisons, strict host flags, raw-input integrity and safe teardown remain.
Fresh pinned application/full-diff proof is under
`target/fused-residual-order-v3`. Native/GPU verification of this narrowed source
remains outstanding at this preparation checkpoint.

After compiler clearance, source commit `7579f98` passed 151 CPU policy checks,
eight Intel strict-host arithmetic cases and twelve separate failure fixtures.
Only `binbcast.cpp` and `fusion.cpp` include the changed inline policy; those two
units rebuild sequentially with the original producer flags. The graph dispatcher
is reused unchanged. Exactly two members change in the 100-member SYCL archive,
and the other 98 members plus engine/base/CPU/registry inputs remain identical.
The completed archive SHA-256 is
`8670c65bd2489294eba9b84604edf5285987b08a744837c2441ed1366bcb8fed`.
Patch 003 is `37b729d6ebbbe0c5e9a0408cb55224e0151de9bf6a32e4e8466ef40bded037ca`;
fresh pinned application/full-diff proof and post-build input checks pass.

The strict wrapped oracle is
`target/fused-residual-order-v3/native-v2/sycl-mul-add-arithmetic-test`, SHA-256
`16ff65601811f0cb99eeb571d98af86871193c9e05912c23833ebf107c2281db`
(84,668,664 bytes). Receipt `native-v2/build-receipt.json` has SHA-256
`0219516e18fac78bbc538bf4a0e5502f8efd9fda8cbe7798150c3ee72f787de0` and records
all wrapper symbols, exact commands, strict host driver flags, CPU outcomes and
archive-member proof. The builder corrected two receipt-driver issues: a driver
dry run needs its borrowed library paths to exist, and expected changed members
must be compared as a set while preserving the archive's original order. These
corrections change no source policy, arithmetic test or native compiler flag.
All earlier artifacts remain intact. GPU execution, a new private worker/runtime
and installed selection remain unperformed; the owner requires all twenty exact
device cases and observed routes to pass before another worker build.

The owner subsequently reports all twenty protected `16ff6560` device cases
passed, including exact special-value/alias comparisons and rejected reverse
legacy routing. The run exited normally with no new kernel fault. After this
gate, an independent copied Cargo target invalidated only `kuno`/`kuno-image`
project artifacts, then rebuilt both crates with one job and the unchanged
reviewed private-worker producer/linker flags. The private worker contains no
oracle wrappers or routing-counter symbols. Rust production source, Cargo lock,
engine `56fc0401`, GGML base/CPU/registry and all prior producer artifacts are
verified unchanged; SYCL archive `8670c65b` is the narrowed candidate above.

The private ABI 2 worker SHA-256 is
`6a714b9732aabe21d44d92fe95e868c1e950d22eee12d5b58c6814cbf265093a`
(135,528,976 bytes), under `native-v2/cargo-target/release/kuno`.
Flat closure `native-v2/runtime-flat` copies all 63 vendor library/notice members
unchanged from the accepted resident closure. Its manifest SHA-256 is
`249f283ef9b67ce4f48f5fde1bb2aacd57a33d5fa3f79f623ea79ce59d191ea0`.
The unselected `native-v2/experimental-runtime.tar.gz` is 156,237,514 bytes,
SHA-256 `94f8f514aa9326e701676a21e86304d9dd399630f41e94e703f66a4cb6ccdc78`.
Packaging uses the unchanged helper with an in-memory candidate-local pin map;
no production packaging pin is advanced.

Worker receipt SHA-256 is
`63270200f06f8b8f34b24a5a87586fccb751466ae8195f42466215db99c27501`;
runtime receipt SHA-256 is
`621a79035c14f0546f9456bac71658b7ad4e1bd0eaf73e7e4a09c505045f287b`.
The complete public engine/GGML source inventory has tree SHA-256
`3124d322455368c532abdfde4af98a74091a1d088d184739b5f66d1bf4539141`,
bound in `native-v2/source-proof-receipt.json`, SHA-256
`26575269b02e6ebaed1e675d3b731e859d5bc5d6517e3908770c78496e6ee486`.
This builder executes no worker/GPU code or service/installation action.
Root-owned matched workflow/pixel/performance and lifecycle validation remain
required before selection. Compare with the f6 overlay to isolate fusion effects;
the older accepted baseline remains necessary to resolve f6 fidelity separately.

### Expanded finite coverage and device fatal boundary

Test-only commit `d261245` preserves the original twenty device cases and appends
twenty-eight finite cases. Public deterministic bit patterns supply full F32
mantissas, bounded TANH-range coefficients, every finite exponent encoding,
subnormals, product/addition halfway ties, cancellation and FMA-sensitive triples.
The three families run both ADD orders, small exact aliases and 4096x9 shapes;
the bounded family also runs both public 4096x4056/8214 shapes. The original
twenty cases retain their former ordering and output. Appended pass lines add
only the fixed numeric family IDs 2 (bounded), 3 (wide) and 4 (rounding).
Every device comparison still requires exact bits, unchanged inputs and actual
eligible/fallback routing. Unsupported reverse orders never force the helper.

The test-only `binary32-reference.hpp` rounds integer significands to nearest
even at the two former binary32 boundaries. It handles finite input products,
signed zeros, subnormal/normal carry and overflow infinity plus a finite
residual; NaNs and two infinities are deliberately outside its scope. Its
standalone strict-host test passes 58 known bit-result/boundary fixtures and
250,000 public mixed-exponent triples, including 6,048 FMA-distinct results,
rounding in both directions, subnormal inputs and overflow products. The graph
test passes twenty CPU cases, cross-checking the integer and strict-host
references for each new family. A separate generator fixture proves its
determinism, both rounding directions, all finite exponent encodings in the
wide family, bounded gates and cancellation/tie controls. Independent source
review found no defect within the declared finite nearest-even scope. These
bounded checks do not exhaust binary32 or formally prove a handwritten oracle.
GPU acceptance compares the old/direct/real-graph outputs independently of this
reference; FTZ/DAZ or any other bit difference remains a failure.

The owner reports the expanded v4 device run passed all forty-eight exact cases
in 13.677 seconds without a new kernel fault. This establishes its clean-path
numerical/routing checks only. A subsequent review found that a thrown backend
call could unwind host upload/readback vectors before the top-level catch,
despite the old destructor guards skipping uncertain device frees. The v4
artifact remains preserved under `target/fused-residual-order-v3/oracle-finite-v4`;
its failed-device cleanup is superseded, with no failing path observed in that
reported run.

Test-only commit `56986ab` makes device backend uncertainty terminate immediately
at the call/status boundary. Device mode is enabled before discovery or
initialization. Exceptions, an already-unsafe entry, failed submission/status,
or uncertain synchronization/release emit only the fixed phase diagnostic and
call `_Exit(2)` before caller-owned host vectors or device resources can unwind.
Confirmed numerical failures keep synchronization and normal cleanup. CPU mode
retains its original throws and cleanup fixtures. Nine new fork/pipe CPU
witnesses cover thrown/false initialization, query, thrown/false submit, failed
compute status, sync, free and unsafe entry; each requires exit 2, the exact
fixed diagnostic and a start-only marker with no host destructor or retry.
All twenty CPU graph cases and twenty-two fixtures pass. Independent source
review found the backend sites covered. Successful pinned `wait()` still does
not establish detection of deferred errors, and the test cannot intercept an
internal backend exit.

The corrected strict oracle is
`target/fused-residual-order-v3/oracle-finite-safe-v5/sycl-mul-add-arithmetic-test`,
SHA-256 `99fe242e0c31e2407eb274e3123132cd640c2c5e0b09f15b45c4b126b848d4a3`.
Its receipt SHA-256 is
`e7f9fe7663ade7cf88515359d301a2a113f217232111ccd58f69164917131b08`.
Actual host compiler flags retain strict rounding/exception behavior and
contraction disabled. Engine `56fc0401`, SYCL `8670c65b`, all borrowed archives
and the exact pinned source/patch proof remain unchanged. The builder executes
no GPU or worker and holds compilation during owner timing windows. The fusion
and f6 overlay remain unselected; the installed runtime is unchanged.

For a future accepted-RoPE plus guarded-SDPA/residual candidate, source must
instead contain accepted patch 001 `ac51db708fef026fe7abab850774309e4f1f439a05c56d06ff33e49fcc290584`
with current patch 002 and narrowed patch 003. The independently rebuilt
accepted-001 engine `f900c54ef095ae2f5899d03e7241befba9bc069320e8c5501d7050024249e607`
can replace `56fc0401`, while this unchanged `8670c65b` backend and matching
base/CPU/registry archives can be reused after complete producer/source proof.
An isolated source/patch tree must pass the normal build verifier; a fresh
private Cargo rebuild and ABI 2 closure must bind those exact inputs. Such a
combination has not been built or validated here and needs owner workflow and
lifecycle acceptance. No standard CUDA/Vulkan RoPE change is required by these
SYCL-only backend optimizations.

## Bounded acceptance and related audit

After the owning task clears compilation, build/run the CPU tests with one
compiler job, then compile changed SYCL translation units and the direct oracle
against exactly the candidate archive. The owning task performs GPU tests. Exact
old/direct/real-graph outputs must pass before warmed timing at `4096x4056` and
`4096x8214`, with uploads/allocations/readbacks excluded. Compare synchronized
old MUL+ADD and eligible fused graph execution, then measure the unchanged full
generation/edit workflows without concurrent compilation/inference. Keep the
installed runtime unchanged until the complete candidate wins and existing
cleanup/cancellation/reuse/privacy acceptance passes.

Q/K/V already share their INT8 activation pack (`ggml_block.hpp:250`); duplicate
activation packing is not a new gain. GPU prefix caches are directly returned
by `GGMLRunner::to_backend`, while host cache staging is CUDA-only. Graph-cut
captures on SYCL apply only when memory pressure forces segmentation. Source cannot
establish that a full-workflow run segments; numeric phase counters would be
needed before pursuing cut-buffer ownership changes. Those changes have greater
lifetime risk than this allocation-free fusion.

Pinned ComfyUI `9d3bfc393b3415e146efb8434b32784cc2e69521` uses fused normalization/
modulation and gated residuals in its
[Qwen21 model](https://github.com/Comfy-Org/ComfyUI/blob/9d3bfc393b3415e146efb8434b32784cc2e69521/comfy/ldm/qwen_image21/model.py).
Its [XPU INT8 capability predicate](https://github.com/Comfy-Org/ComfyUI/blob/9d3bfc393b3415e146efb8434b32784cc2e69521/comfy/model_management.py#L1905)
returns false, so its reported warm time does not establish an identical native
INT8/accumulation path. This experiment preserves the current native arithmetic.

ConvRot's double rotation remains a separate candidate: a per-group F32 scratch
then row maximum/pack can remove half the transform work, but adds two launches
and changes approximate activation global traffic from 9 to 13 bytes/element.
Extra scratch including group maxima is 135,103,872 bytes at `8214x4096` and
405,311,616 bytes at `8214x12288`. Legacy SYCL pool allocation can abort on OOM;
an optional scratch fast path therefore needs a checked cap and a non-aborting
allocation before any submission, accounting for retained pool highwater.
The existing ConvRot test only covers small `K<=512` with tolerant scales;
stage changes require exact packed-byte/scale-bit/padding comparisons and large
public K plus nonfinite/ties/subnormal cases. No quantizer change is included.
