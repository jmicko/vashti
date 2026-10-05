# Selected managed SYCL source and package recipe

The managed runtime's public packaging pins select engine patch001
`ed75739ccb193099f23d4600cc4a8846f9d17fe6c2cec7591a7c9cb30c2d0134`,
unchanged patch002 `7298590faed44cad4733c78f6f6c2bf261b647712b5ec5f75840da450f0067dc`,
and patch003 `3d1191da4eb004a25c52116fa30421c3a7b617fc4e854d416248b2d7b0070794`.
The engine/GGML revisions, family ID, worker ABI2 and GGML name160 are unchanged.
Selecting these source pins is distinct from publishing a release.

`build-image-runtime-sycl.py` is the normal managed worker build entry. Supply an
independent, completely patched source checkout at those revisions and the
reviewed oneAPI 2026.1 bundle. Unknown source diffs, untracked or ignored files (including CMake GLOB inputs),
pin mismatches and existing native/Cargo output directories are rejected. It configures and compiles with
one job, then uses the ordinary Cargo source/cache/ABI verifier and native-image
feature. It never initializes a device, executes a worker, packages libraries,
installs a runtime or publishes anything.

```bash
python3 apps/kuno/scripts/build-image-runtime-sycl.py \
  --source "$KUNO_SD_SOURCE_DIR" --oneapi "$KUNO_ONEAPI_ROOT" \
  --mkl-config "$KUNO_MKL_CONFIG_DIR" --output "$KUNO_NEW_BUILD_DIR" --plan
```

Omit `--plan` to build after the plan and inputs have been approved. Plan-only
mode validates the complete three source diffs and prints exact commands and
missing inputs without creating output, running CMake or compiling. A reviewed
`MKLConfig.cmake` is required for a real configuration. The tested closure's
historical MKL cache path was under cleared `/tmp`; it cannot substitute for a
new reviewed build input. No fresh CMake configuration/native build is claimed
by the public selection checkpoint.

This managed builder explicitly passes `-DKUNO_SYCL_WHOLE_ROPE_BYPASS=ON`.
The source condition remains unset/OFF by default, and the standard CUDA/Vulkan
builder is unchanged. Release C and C++ flags are explicitly `-O3 -DNDEBUG`,
matching the tested two whole/dispatcher units and all 57 reused engine C++
commands; there is no O-level mismatch. Whole's existing source CMake rule adds
precise/no-fast-math/no-associative-math/contract-off options. No TEST, repeat
bypass, profiler or link wrapper is enabled. Inherited C/C++/link flags and
CARGO_ENCODED_RUSTFLAGS are cleared before the fixed native settings and reviewed
Rust linker are applied; an environment-only CPU fixture covers precedence. Name160, FP16, oneDNN, static engine
and archives, graph OFF and host fallback OFF match the tested configuration.

Fresh CMake-generated commands still need comparison with the tested producer
commands, including shared-export visibility/PIC, compiler/package detection and
Level Zero discovery. The tested native closure reused proved archives rather
than claiming a fresh full CMake build. A newly compiled worker requires renewed
exact numerical, full-workflow, SDK-hidden, privacy and cancellation acceptance.
The tested worker can be installed separately under the owner's authorization.

Normal packaging now accepts the tested flat closure directly:

```bash
python3 apps/kuno/scripts/package-image-runtime.py \
  --directory "$KUNO_APPROVED_RUNTIME_FLAT" \
  --manifest-sha256 fc169dd2bf4418c2770940c85ec570f5b0a8c5ddaa21e32176cf839b36c88baa \
  --output "$KUNO_NEW_RUNTIME_ARCHIVE"
```

The direct selected-pins check produced archive SHA256
`85d7aecb92e9e27af85f56bf5569d31feffd63d37955e7549ac081ab3e3e54a3`
(156,247,147 bytes), byte-identical to the tested private package, with no in-memory
pin override. The old packager helper/private script and receipts are preserved
under `target/whole-rope-residual-v1/public-selection-v1`; six native inputs,
worker, 63 vendor members and the old archive remain unchanged. The new archive
is a local packaging proof; this helper does not install or publish it.

The parent reports the counter-free B70 matched comparison passed eight full
requests with no new Xe/privacy/PID/shutdown failure. Whole cancellation and
fresh two-step edit recovery passed. Warm generation was 132.893→106.523seconds;
warm edit 72.769→61.831seconds. These are owner-run tested-worker measurements,
not measurements of this unexecuted fresh build recipe. Independent same-worker
edit variability controls are separate from selecting source pins.

Global patch001 changes Qwen21 graph construction across all backends. Rebuilding
and accepting the standard CUDA/Vulkan embedded engine and standard native-image
gateway remains a release gate. The standard package script's no-private-Intel
linkage/RPATH check and normal release checks remain required and unchanged.
There is no C++/Rust/API/UI, precision, requested workflow, source-default or
other-backend build-setting change in this script/package selection commit.

The final selection receipt is
`target/whole-rope-residual-v1/public-selection-v1/selection-receipt-v2.json`,
SHA256 `9a0003e56434665880119e4bcea7aedd5d034e9ab223db85fc9e8d769a24aefe`.

Run `python3 apps/kuno/scripts/test-build-image-runtime-sycl.py` for six CPU-only
source/environment fixtures; it invokes only Git on synthetic local repositories.
The guard rejects both untracked and ignored engine/GGML files before plan or
configuration, while allowing reviewed intent-to-add files represented in the
complete patch diff. Source-guard receipt-v3 SHA256 is
`959c382f57fd7f02339e9516950bf0f7eb7184a26c2dfaf8a896287872d45fd5`.
