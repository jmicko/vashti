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

Intended complete validation adds seven application CPU controls, seven runtime
controls, 176 compiler layouts (18 regular plus4 grouped definitions, each with
8 layouts), and104 paired numerical controls (68 regular plus36 grouped) to the
previous326 application CPU,34 full runtime and573 GPU controls. The original
feature-off controls, separate eight metadata controls, shader populations and
both original18-script whole-model chains remain. Existing public numerical
oracles, source/owner/fault checks and tolerances are unchanged. Compilation and
GPU results are pending; these planned counts do not establish acceptance.

The complete bundle is measured only after correctness and physical recovery
checks. A diagnostic speed result does not by itself establish Strata parity or
authorize installation. Root owns integration/builds/GPUs; isolated agents only
edit source. No install, release, reset, reboot, or frontend change is included.
