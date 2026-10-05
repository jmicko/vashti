# Private whole-RoPE and narrowed residual composition

This isolated branch is based on preserved fusion `58c6bdc`. It combines the
unchanged narrowed residual/guarded attention backend `8670c65b` with the
default-off whole-RoPE source from `cc028e5`. It uses the f6 shared prepared-PE
engine patch001 `ed75739c` and engine archive `56fc0401`, not accepted001/f900.
Its only valid comparison baseline is the existing `qwen21-fusion-order-v3`
private closure. The validated accepted-RoPE candidate remains separate and
unselected; installed and release paths are unchanged.

The complete SYCL patch003 digest is
`3d1191da4eb004a25c52116fa30421c3a7b617fc4e854d416248b2d7b0070794`.
Engine and GGML revisions remain `88411ef1e0688ff2df1010aeeb5d92b2d8cea2be` and
`4bf5f6000653b7881d00963cd6ddb665ccd62a8d`; patch002 is unchanged. Fresh
independent source/proof clones reproduce all complete patch diffs. The normal
Cargo source verifier is preserved; added rerun paths cover the new headers and
existing residual sources. New files are represented in the native Git diff.

Only whole-RoPE capability initialization, graph preplanning and per-node step
hooks are added to the residual dispatch. Whole step runs before existing
fusers. `binbcast.cpp`, `binbcast.hpp`, `fusion.cpp` and `mul-add-policy.hpp`
remain byte-identical to backend8670. All five whole files and the span helper
`repeat-mul-policy.hpp` match the native-tested whole source byte-for-byte.
Inherited repeat-bypass dispatch and kernel are omitted. Whole's private macro
is OFF by default; production builds may enable WHOLE_BYPASS but must omit
WHOLE_TEST, repeat, profiler and test-wrapper definitions.

The independent merge/reuse review receipt is
`/home/jmicko/.codex/worktrees/kuno-convrot-local-review/vashti/target/whole-rope-residual-merge-source-review-v1.json`,
SHA256 `5555b38b3e6b573451b212fb83b74c03a60a77d741ec744ef1145a62416de82d`.
It verifies hook ordering, disjoint whole/residual shapes, original producer
lifetime/use guards, 57 producer C++ dependency files covering 211 public
sources, full GGML headers and engine56fc/15 unchanged C members. This is source
provenance, not merged runtime acceptance.

The owner's isolated whole source passes 84 native cases/1,392 exact vector
comparisons with replay, changed PE, rebound pointers, special/wide words and
capacity fallbacks. Test-only `d5d0b1c` appends actual Qwen RMS→MUL_INPLACE
producer fixtures at L65/L8214 after those original cases: expected native
86/1,400 and CPU79/1,372. Production guards/kernel are unchanged. The combined
TEST backend must pass the expanded whole oracle and unchanged residual48
oracle under owner GPU control before a separate counter-free worker is built.
All uncertain device failures terminate before host-buffer unwind; no oracle
asserts stronger deferred-error detection than pinned backend synchronization.

Actual full-model routing still needs measurement. Shared prepared PE does not
force its CONT before the first monolithic activation window under GGML DFS,
so that first window can decline. Cached/prelude segments may expose ready PE;
in-place normalized input owners are now covered explicitly. No guard is
relaxed to force routing, and isolated graph timing is not full-image timing.

Artifacts and receipts are under `target/whole-rope-residual-v1`. Reuse of native
inputs requires matching all source dependencies and flags; production replaces
only the proved changed object members in one final SYCL archive. The private
worker must use a fresh independent Rust target with `kuno`/`kuno-image`
artifacts invalidated, normal verification, ABI2/name160, no counters or wrappers,
and 63 byte-identical vendor members. The owner must validate its SDK-hidden
closure, full workflows, request cleanup, privacy and cancellation/recovery.
No installed runtime, API, standard CUDA/Vulkan source or release pin changes
are authorized by this experimental branch.

At source checkpoint `230d15b`, the merged TEST backend and two native oracles
are compiled under `target/whole-rope-residual-v1/test-native-v3`. CPU-only
executables pass 79 whole cases/1,372 vector comparisons, nine failure fixtures
and actual-engine56fc request/cache cleanup. Canonical CPU registry/link glue
avoids GPU registration and initialization; CPU executables have no SYCL, CUDA,
Vulkan, Level Zero, OpenCL, MKL or oneDNN dependency. Host compilation is strict,
with active assertions. Initial harness attempts are preserved: the first
compiler command rejected FP-option ordering, and the next failure-fixture
invocation omitted its required numeric argument. Neither was a numerical
mismatch. Native compilation then passed; strict host-cc1 parsing was corrected
to distinguish the x86 host triple from the device command's x86 aux-triple.

The merged TEST archive contains 101 unique members. Only `ggml-sycl.cpp.o` is
replaced and `whole-rope.cpp.o` added; all original members, including residual
`binbcast.cpp.o` and `fusion.cpp.o`, match backend8670 byte-for-byte. Both oracle
host cc1 commands have strict rounding, no unsafe/FTZ/DAZ options and no
RPATH/RUNPATH. The whole binary intentionally has two TEST control symbols; the
residual oracle additionally has its three scoped link wrappers. They are
validation artifacts and cannot be used as production worker inputs.

| Artifact | SHA256 |
| --- | --- |
| `source-receipt-reviewed.json` | `d9cf57be85ae7e47064409cb3e40391a9714f92e8fe1c05b0446a30f4ed31767` |
| `test-native-v3/libggml-sycl.a` | `ed8968b400929d370bc57c98e760ebbdb8056ec1ae527be644922c85bd5c923d` |
| `test-native-v3/whole-rope-test-sycl` | `075623cb6dca957c1a330b47dbd3cb42d034de614cc8c49c8cd18977e591e500` |
| `test-native-v3/residual-arithmetic-test-sycl` | `2ebf62527750ecf08cc6f8bb29ba1500427b7f8c3043eb953185c90604c590f6` |
| `oracle-build-receipt-v4.json` | `76c25dc73a7a9c925f8ac32d2253046f6a89054de451fc7d82e2ddd80c956726` |

Each native oracle takes `--device SYCL <canonical PCI>` under the protected
owner harness. Expected exact counts are whole86/1,400 and residual48. The
builder executed neither native binary. All compiler/copy jobs are idle at this
checkpoint. A separately compiled counter-free backend and worker remain held
until those merged device checks pass and the owner clears the next build.

At the production build checkpoint, the owner reports both merged GPU oracles
passed: whole86/1,400 with 127 selected/launches and 34 produced fixtures,
including actual in-place cases85/86, and unchanged residual48. Both exited
normally with no numerical/routing/input mismatch, device uncertainty or new Xe
fault. The build agent did not execute a GPU binary.

A separate counter-free backend compiles only the two changed units. Its final
archive has 101 unique members, preserves the remaining 99 baseline8670 members,
and contains neither test/profile/repeat counters nor wrappers. The worker
build uses the ordinary complete source verifier, independently copied target
cache and explicitly cleaned Kuno/Kuno-image project artifacts. Cargo clean and
build passed. Cargo's release output has its normal release/deps hardlink, so a
separate regular nlink1 standalone worker is verified byte-identical. The first
finalizer attempt and its successful Cargo outputs remain preserved.

The private flat ABI2/name160 package contains that worker and 63 byte-identical
accepted vendor/library-notice members. The normal package helper is unchanged;
its source-proved private pins are supplied explicitly in memory while selected
release pins remain unchanged. No worker/package was executed by the builder.
All compiler/copy/package jobs are idle. No API, UI, requested recipe, precision,
private persistence, installed service, standard CUDA/Vulkan or release changes
are made by this closure. Full workflow acceptance remains with the owner.

| Frozen artifact under `target/whole-rope-residual-v1` | SHA256 |
| --- | --- |
| `production-native-v1/libggml-sycl.a` | `277b639c2c912ba815b4250303b8e6f1cf17f47df7d40270c345ac03694730e0` |
| `standalone-worker` | `4deb9a1bcd7108be8d22b56e8cc8958cb52458a56361c3b19c2b193012e679a3` |
| `runtime-flat/manifest.json` | `fc169dd2bf4418c2770940c85ec570f5b0a8c5ddaa21e32176cf839b36c88baa` |
| `experimental-runtime.tar.gz` | `85d7aecb92e9e27af85f56bf5569d31feffd63d37955e7549ac081ab3e3e54a3` |
| `worker-build-receipt-v2.json` | `4bda8a479d62e6d3617645328605c88aaaab07986a0cdab1c7397b2e32751148` |
| `runtime-receipt-v1.json` | `e78c2e1d8c7cdb0f72b0fe70185be903ff5bd5137f1bba5ee4171e65f790512e` |
| `READY-v1.json` | `c6b48f1b15f5d926498787927f3b8c1517a52bf1f2e03f8fae82c01e8b060c33` |

The reproducible activation is the ordinary CMake option
`-DKUNO_SYCL_WHOLE_ROPE_BYPASS=ON` (default OFF), compiling production with
`-DKUNO_SYCL_WHOLE_ROPE_BYPASS=1` and no TEST/profiler/repeat definition.
Whole's CMake rule retains precise, no-fast-math, no-associative-math and
contract-off flags. `production-native-receipt-v1.json` records the exact
compiler argv arrays and preserved source/member hashes;
`target/whole-rope-residual-v1/rebuild-recipe.md` describes native reuse, fresh
ordinary Cargo verification and the explicit private package pin substitution.
