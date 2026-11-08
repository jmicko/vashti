# Small-expert direct publication

This internal, optional `strata-small-direct-publish` bundle extends the accepted
expert-ordered output contract to macro32/macro64, narrow8/narrow16, grouped
experts, and actual wide-kernel tails. It is composed with raw GU activation,
ordered DOWN inverse gathering, and the existing grouped expert scheduler.
It changes no API, model format, privacy policy, or Vashti behavior.

The active pinned MIT Strata Intel path stores completed matrix fragments to
its global GU/Dm destinations. Kuno's previous smaller producers instead wrote
those fragments to shared memory and scattered them with scalar threads. The
new genuine producer entries retain their original operand staging, chronological
K16 arithmetic, matrix geometry, six bound owners, shape guards, and fallback.
For each complete eight-member fragment, a real subgroup vote checks all eight
routes before a cooperative global store. GU is raw and must pass the existing
finite-check/activation phase. DOWN writes expert order and pending status2;
the existing inverse/finite consumer resolves it. A partial or invalid fragment
uses the original checked scalar publication with subgroup barriers around its
exclusively owned scratch region. No conditional workgroup barrier is added.

Regular entries use `strata_matrix::native::small_direct`; grouped entries use
`strata_matrix::cohorts::small_direct`. Native grammar recognizes those exact
identities only when the runtime feature is enabled. Full kernels retain their
original scalar ABI; wide tails retain the actual seven-scalar ABI. A genuine
original wide prefix plus a new tail is allowed only with the correct GU checker
or ordered DOWN consumer. Rank-ordered and expert-ordered tails cannot mix.
Original definitions, entry points, controls and fallback paths remain available.

Counters retain their meanings: native GU checks count all actual groups;
direct-GU prefix counts remain wide-prefix counts. Additive selected-launch
counters identify the small direct kernels. Eligibility alone is not execution.

Complete validation adds eight application CPU controls, seven runtime
controls, 176 compiler layouts (18 regular plus4 grouped definitions, each with
8 layouts), and104 paired numerical controls (68 regular plus36 grouped) to the
previous326 application CPU,34 full runtime and573 GPU controls. The original
feature-off controls, separate eight metadata controls, shader populations and
both original18-script whole-model chains remain. Existing public numerical
oracles, source/owner/fault checks and tolerances are unchanged.

## Completed acceptance, 2026-10-23

Frozen source `4cc3f68f030a859ab36c15b6dad4547ba1da0292`, worker
`19d3606e258037e027563b4f299bfdff5d9cc717d0af54a8e24acfb52cbce7c0`
passes334 selected application CPU controls,18 feature-off and41 full runtime
controls,8 separate original metadata controls, all original compiler
populations and176 added layouts. All677 GPU controls pass, including all573
previous controls,68 regular small-direct and36 grouped small-direct controls.
The21 deep-only64KiB controls remain explicitly unavailable on this48KiB device.
Both unchanged18-script whole-model chains pass with original routing totals
[7663,26673,26673], actual new selection counters and physical recovery.

The selected complete implementation is271d261a,7dbadaac,89b781fe,95134b22,
5f5f2358,1ca4e84e and4cc3f68f. Three failed preparation attempts are retained:
an obsolete source-text fixture required updating for the explicit old/new
selector; the real HAL needed to insert a typed264 subgroup memory-semantics
constant for global-operand kernels which have no workgroup barrier; and the
new offline validator needed the actual selected standard-uniform-layout flag.
The HAL change adds no barrier and retains rejection of malformed mixed scopes.
An additional CPU test checks real emitted global/cohort modules, untouched
instruction words, idempotence and ID-bound overflow. No numerical tolerance
or original ownership check was weakened.

Diagnostic first/repeated long prompt times are11.004590487/8.946788843s in
profile mode and10.967607646/8.854793404s in host-profile mode. The latter is
slightly slower than preceding10.134471518/8.745485045s; it is not evidence of
clean parity or a reason to discard part of this coherent producer/consumer
change. Matched layer24 matrix time rises36.353866->39.483661ms, including
cohorts10.839204->12.709929ms. Dequant19.224670ms and copy29.172595ms remain.
All8 actual wide programs and all new small programs have zero spills; the two
older unrelated spilling program hashes and original QSA ISA remain unchanged.

Evidence under `kuno-development`: `small-direct-root-build-receipt-20261023.json`,
`strata-parity-small-direct-20261023-expert-cohorts-public-summary.json`,
`small-direct-{profile,normal}-safe-phases-20261023.json`,
`small-direct-complete-isa-export-20261023.json` and
`small-direct-final-state-20261023.json`. The protected service exited0 in
11m46.908s, peak swap0; Intel recovered to50,688,000B, NVIDIA compute0, all five
kernel fault counts0, admission normal and Kuno/Ollama/Vashti HTTP200.
The original clean45/4245/4245 comparison passes with first/repeated long
12.471177197/8.394039699s, versus Strata5.5006/5.3515s. The goal remains unmet
at both positions. Receipt: `strata-parity-small-direct-20261023-fixed-clean-public-summary.json`.
This separate uninstrumented run exited0 in2m21.776s, swap0 and physical
recovery passed. Four output tokens do not establish sustained decode speed.
No installation was performed.

The complete bundle is measured only after correctness and physical recovery
checks. A diagnostic speed result does not by itself establish Strata parity or
authorize installation. Root owns integration/builds/GPUs; isolated agents only
edit source. No install, release, reset, reboot, or frontend change is included.
