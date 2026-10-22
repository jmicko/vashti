# Native-only compute preparation

Optional backend-only candidate `strata-native-prepared` / `--native-prepared`.
It requires the coherent Intel core but is independent of `strata-compute-main`;
RCS and CCS candidates can select it separately. No API, protocol, model,
configuration, frontend, installation or release change is included.

The original native path creates an ordinary Wgpu compute executable to obtain
its host identity/registration, then creates another actual Vulkan executable
in the native graph. Native graph errors already return errors; they never fall
through to ordinary inference. This candidate omits only the unused ordinary
`vkCreateComputePipelines`, retaining genuine Wgpu root initialization and the
complete unselected ordinary path.

## Complete coupled contract

- Both ordinary creation and native preparation share original core resource
  resolution: Fallible IDs, valid device, passthrough-without-layout rejection,
  actual module/layout/cache device checks, compute support, finalized entry,
  full original shader/interface/binding/workgroup checks, explicit or derived
  layout creation and original late-sized buffer information.
- Original shader-module creation remains. Native preparation calls the exact
  original Vulkan `compile_stage`, including lowering, override processing,
  runtime/binding options, workarounds, canonical workgroup/FP16/subgroup rules
  and original error mapping. It destroys only newly created temporary shader
  modules, never borrowed shader handles. The graph still performs actual
  native workgroup/subgroup/shared-source admission and creates the real driver
  executable. No weaker claim that core interface alone validates shared SLM.
- Native options currently require empty pipeline constants and workgroup
  zero-initialization disabled, exactly as original CubeCL launch construction
  supplies. Selected native WGSL rejects override declarations. Ordinary mode
  keeps its original full option semantics. Malformed raw entry NULs return a
  typed preparation error rather than entering the original CString unwrap.
- The opaque prepared object retains its exact Device, ShaderModule, core
  module/layout/admission and finalized entry. Module/layout/cache contexts are
  checked before IDs are extracted. Numeric IDs or matching source bytes do
  not substitute identity across different Globals. The native factory checks
  the exact original device and module; Graph checks actual device provenance.
- Automatic layouts remain opaque: there is no layout/BGL getter or ordinary
  encoding interface for a native token. Ordinary creation still finalizes its
  original automatic-layout exclusive-pipeline association unchanged.
- An exhaustive typed native/ordinary pipeline enum travels through the server
  cache, scheduler, expert batch and Graph. Native identities cannot wrap or
  reuse reserved zero. The original consumptive module registration remains;
  no new module/source cache or reuse policy is introduced. The real native
  graph owns the admission Arc through teardown, alongside its original shader,
  layout and executable. Original actual VkPipeline replay identity is unchanged.
- Kernel identity includes the same execution mode, native release boundary
  and external-source mask. Bindings, INFO, request scopes, physical reads,
  source/allocation owners, root initialization, dispatch, completion, quarantine,
  erasure and actual buffer hazards remain on their original paths. A native
  token is rejected if it reaches ordinary dispatch; errors do not trigger an
  unowned fallback. No request data is retained in a prepared pipeline.

## Validation and remaining acceptance

CPU controls cover real identity exhaustion/zero handling and validate that the
fixed-public shader fixtures are genuine valid Naga IR before device admission.
The fixed-public matrix mode adds eleven actual-device admission controls.
Eight compare prepared success/failure directly with original ordinary creation:
explicit and automatic layout, missing entry, wrong/missing binding, actual
workgroup limit, passthrough implicit-layout rejection and explicit-layout
success. Other controls check exact module identity and both unsupported native
stage options. Comparison-only ordinary pipelines are not used to prime or time
whole-model speed fixtures. No private input, buffer or dispatch is constructed.

Existing original CPU/runtime/compiler/layout controls, all241 GPU operators,
both complete whole-model profile/count/source/owner/erasure/recovery chains,
original terminal failures and clean first/repeat selection remain required.
Native-prepared public process counts report prepared admissions, ordinary
executables and actual graph-program creation by key kind. Root initialization
is deliberately outside these counters because it remains genuinely ordinary.
Whole-model selected counts must show prepared/native graph programs and no
ordinary inference executable or original-key graph program.

At source checkpoint the code is not installed. Selected RCS/native-prepared
compiles and all183 CPU controls pass (original179 from compute-primary source
retained, plus four new controls). Unselected original RCS also compiles and all
179 CPU controls pass. No protected GPU acceptance or speed result is claimed.
Wrong-context/device rejection has explicit provenance guards, but a new second
physical device/context negative control has not been run. Root alone owns
hardware tests, service changes and the eventual frozen artifact/installation.

## CPU evidence and root handoff

Implementation starts from69c045a7 in the separate managed worktree
`codex/native-prepared-pipeline`. Root alone will merge/freeze the source and
run the original protected chain with RCS plus this option; compute-primary is
not implicitly selected. The shared Cargo target was used only during the
explicit exclusive CPU phase, with two build jobs and no hardware operations.

Evidence under `/media/jmicko/faststore/kuno-development`:

- `native-prepared-cpu-{artifacts,build,controls,selection,features}-20261022*`:
  selected183 CPU checks pass, complete feature list and exact test names.
- `native-prepared-unselected-cpu-{artifacts,build,controls}-20261022*`:
  original mode179 CPU checks pass. One harmless unused-import warning exists
  when the optional native-prepared variant is absent; no runtime warning.
- `native-prepared-cpu-*-attempt1-20261022*`: first compile failed on an added
  error variant using ShaderStages instead of ShaderStage; fixed before tests.
  No GPU was invoked. Attempt2 compiled before final registration cleanup;
  final selected/unselected checks above cover the actual resulting sources.

No source under `vendor/cubecl-runtime` changed. No CPU result substitutes the
original runtime, offline421 WGSL/248 matrix modules,241 operators, eleven new
actual-device admission controls, complete owner/source/proof/recovery chains,
terminal controls, clean selection and staged quality/API installation gates.
The retained counters and public controls contain no user prompts or images.
