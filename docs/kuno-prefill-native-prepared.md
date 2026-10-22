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


## Frozen root acceptance checkpoint

Integrated source301f1fd16f02093be7882db066f276555bfe3245 (portbef9ef50), artifact
`strata-parity-native-prepared-20261022`, SHA256
0e22682700fcf269d5cd3df1b321622c2b62bc422bbda077b667d0c3443ce6de.
Root exact-source183 CPU controls PASS again, original runtime tree unchanged,
421 WGSL/248 matrix layouts PASS pinned2026.1 validator. Render-primary selected;
complete compute-primary optional path remains. Two independent source reviews
find no blocker; all private ownership and actual native program checks retained.

Root protected unit`kuno-native-prepared-20261022.service` now runs original241
GPU operators plus11 actual-device admission checks and BOTH original
73/4245/4245 compiler/count/owner/source/capacity/erasure/recovery chains. No
GPU/speed/terminal/clean/staged/install acceptance is claimed yet. No build or
other GPU test runs concurrently. Installed worker/gateway unchanged.

The original18-script proof chain is frozen separately without mutating prior
helpers, under evidence base`native-prepared-original-harness-20261022/`.
Each predecessor is preserved in`original/` with both source hashes in manifest.
Two profile leaves ADD actual prepared/ordinary/native-key/original-key counter
validation. One compiler inventory guard now requires exactlyONE program per
QSA scope when native-prepared selected, originalTWO otherwise, because omission
of the duplicate executable is the intended behavior. Every original actual
native subgroup32/MAD180/no-spill check, original scalar/F64 references,
numerical thresholds, real group/count/source/cap/owner/replay/erasure/recovery
check remains. No synthetic ordinary executable is created to satisfy oldcounts.
Native admission comparison pipelines run only in separate operator process,
before scoped compiler controls, never in timed whole-model workers.

Frozen manifest SHA256
1f328ab33d36bd3006d2dc76c3d0362a42c72e8d0e111f50a7a580a4eed97f88.
Connected wrapper`check-native-prepared-connected-20261022.py` SHA256
d7ddeb8b57bde06dc00f3e50c34b331f3408800d1d8ac0c585fd9baef09c91c9
pins exact original/new helpers, source/artifact/features/all183 CPU names,
offline inventory and original11 runtime checks. It requires all11 exact
admission cases, original241 operators and three actual selected-program counter
rows in EACH full profile mode, with no ordinary inference executable or original
Graph key. Genuine ordinary root initialization remains independently exercised
by the original numerical/owner/erasure checks. Required Vashti changes:none.

## Complete connected outcome

Frozen301f1fd1/SHA0e226827 passes all183 CPU, original11 unchanged-runtime,
421 WGSL/248 matrix layouts, all241 original GPU operators and eleven new
actual-device admission controls. BOTH complete original73/4245/4245 schedules
pass all actual native compiler, FMA/group/count/source/owner/capacity/erasure/
recovery checks. Exact full public proof:
`strata-parity-native-prepared-20261022-native-prepared-public-summary.json`.
Protected unit completed successfully/inactive/dead. No ordinary inference
executable is created: both modes report261/271/271 prepared admissions and
actual native Graph programs, with zero ordinary or original-key programs.
Genuine ordinary root initialization remains. This count change is verified,
not inferred from elapsed time or absence of errors.

Normal first/repeat10.116598360/9.222042861s versus global-narrow
10.653114762/9.164950382s: first5.04% shorter, repeat0.62% longer. Sampled
first/repeat10.236566655/9.225692768s versus10.653008324/9.261120500s. This is a
modest first-use improvement with effectively unchanged warm performance;
Strata5.3515s remains faster. No sustained decode improvement is claimed.

Original terminal-negative/fresh recovery and clean45/4245/4245 selection now
run under`kuno-native-prepared-acceptance-20261022.service`. Wrapper
`check-native-prepared-acceptance-20261022.py` SHA256
08cd7e4585afc54986de7b1e302d393f8d7db95621180ea8454be8fa32fb2f07
pins this exact artifact and complete proof, all11 admission cases and native
program counts. Existing5%-at-EACH-position comparison against installed3fffc266
is unchanged. Full staged tools/thinking/MAX8/semantic/privacy/API checks follow
only if both clean positions pass. No staged/install acceptance is claimed yet.
No API/frontend/config/model/release/reset/reboot change; required Vashti:none.

## Clean comparison passed; full staged checks in progress

Original terminal-negative recovery PASS. Fresh clean45/4245/4245 yields
first10.897223643/repeat8.712129955s versus installed12.962279132/11.147390805s:
15.93%/21.85% shorter, satisfying the original5%-at-EACH gate. First short
45-token input6.575669069s and startup81.838527466s are recorded separately,
not hidden in the long-prompt claim. No prompt reuse or workload substitution.
Exact selection receipt:
`strata-parity-native-prepared-acceptance-20261022-clean-selection-public-summary.json`.

The first full staged helper failed BEFORE inference because the outer strict
UMask0077 created root-owned0700 test directory and0600 config, unreadable by
the actual nonroot staged gateway. Gateway startup failed; original clean
results are not rerun or replaced. Failed log/receipt remain. Physical recovery
passed; independent state shows Intel50,688,000B, NVIDIA compute0, all reported
faults0, all services200 and normal admission. No installation occurred.

The replacement helper keeps every original tool/thinking/MAX8/semantic/privacy/
API test and timeout unchanged, explicitly assigns ONLY the staged0700 directory
and0600 config to the actual nonroot stage user. It does not make copied config
world-readable or change installed config. Helper
`check-native-parity-stage-quality-private-config-20261022.py` SHA256
9f97cc84ffcb8d9d02a4ee6ded02623dd277cab9aacc6dd6c2ab9041b6734e1f;
protected unit`kuno-native-prepared-stage-fixed-20261022.service` now runs full
checks into`strata-parity-native-prepared-private-stage-20261022`. Gateway
HTTP200 and real frozen worker launch confirmed. No full staged/install claim
until completion, nor Strata parity. Native-only artifacts and clean source
remain unchanged.
